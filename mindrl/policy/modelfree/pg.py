from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

import mindspore as ms
from mindspore import ops, nn

from mindrl.data import Batch, ReplayBuffer, to_mindspore, to_mindspore_as
from mindrl.policy import BasePolicy
from mindrl.utils import RunningMeanStd


class PGPolicy(BasePolicy):
    """Implementation of REINFORCE algorithm.

    :param torch.nn.Module model: a model following the rules in
        :class:`~mindrl.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~mindrl.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            model: nn.Cell,
            optim: nn.Optimizer,
            dist_fn: Type[nn.probability.distribution.Distribution],
            discount_factor: float = 0.99,
            reward_normalization: bool = False,
            action_scaling: bool = True,
            action_bound_method: str = "clip",
            deterministic_eval: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        self.actor = model
        self.optim = optim
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        r"""Compute the discounted returns for each transition.

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        unnormalized_returns, _ = self.compute_episodic_return(
            batch, buffer, indices, v_s_=v_s_, gamma=self._gamma, gae_lambda=1.0
        )
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / \
                            np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        return batch

    def construct(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~mindrl.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~mindrl.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, hidden = self.actor(to_mindspore(batch.obs), state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(ops.Softmax(axis=-1)(logits)[0])
        else:
            dist = self.dist_fn(ops.Softmax(axis=-1)(logits))

        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = ops.argmax(logits, axis=-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()  # TODO: meet issues of batch inputs for multinominal
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses = []
        state = None
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                obs = to_mindspore(minibatch.obs)
                act = to_mindspore(minibatch.act)
                ret = to_mindspore(minibatch.returns)

                def forward_fn(state, obs, act, ret):
                    logits, hidden = self.actor(obs, state=state)
                    if isinstance(logits, tuple):
                        dist = self.dist_fn(ops.Softmax(axis=-1)(logits)[0])
                    else:
                        dist = self.dist_fn(ops.Softmax(axis=-1)(logits))

                    if self._deterministic_eval and not self.training:
                        if self.action_type == "discrete":
                            act = ops.argmax(logits, axis=-1)
                        elif self.action_type == "continuous":
                            act = logits[0]
                    else:
                        act = dist.sample()

                    log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
                    loss = -(log_prob * ret).mean()
                    return loss

                grad_fn = ops.value_and_grad(forward_fn, None, self.optim.parameters)

                def train_step(state, obs, act, ret):
                    loss, grads = grad_fn(state, obs, act, ret)
                    self.optim(grads)
                    return loss, grads

                loss, grads = train_step(state, obs, act, ret)
                losses.append(loss.item((0)))

        return {"loss": losses}
