from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
# import torch

import mindspore as ms
from mindspore import ops, nn,ms_function

from mindrl.data import Batch, ReplayBuffer, to_numpy, to_mindspore_as, to_mindspore
from mindrl.policy import BasePolicy

class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    :param torch.nn.Module model: a model following the rules in
        :class:`~mindrl.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool is_double: use double dqn. Default to True.
    :param bool clip_loss_grad: clip the gradient of the loss in accordance
        with nature14236; this amounts to using the Huber loss instead of
        the MSE loss. Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~mindrl.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            model: nn.Cell,
            optim: nn.Optimizer,
            discount_factor: float = 0.99,
            estimation_step: int = 1,
            target_update_freq: int = 0,
            reward_normalization: bool = False,
            is_double: bool = True,
            clip_loss_grad: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.set_train(False)
        self._rew_norm = reward_normalization

        self._is_double = is_double
        self._clip_loss_grad = clip_loss_grad


    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> "DQNPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.set_train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        ms.load_param_into_net(self.model_old, self.model.parameters_dict())  # 可返回网络中没有被加载的参数。

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> ms.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            return target_q[np.arange(len(result.act)).tolist(), result.act.tolist()]
        else:  # Nature DQN, over estimate
            return target_q.max(dim=1)[0]

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~mindrl.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    def compute_q_value(
            self, logits: ms.Tensor, mask: Optional[np.ndarray]
    ) -> ms.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_mindspore_as(1 - mask, logits) * min_value
        return logits

    def construct(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            model: str = "model",
            input: str = "obs",
            **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :return: A :class:`~mindrl.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~mindrl.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        obs_next = ms.Tensor(obs_next, dtype=ms.float32)
        logits, hidden = model(obs_next, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.argmax(axis=1))
        return Batch(logits=logits, act=act, state=hidden)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()

        weight = batch.pop("weight", 1.0)  #

        obs = to_mindspore(batch["obs"])
        obs_next = to_mindspore(obs.obs) if hasattr(obs, "obs") else obs
        act = to_mindspore(batch.act)
        returns = to_mindspore(batch.returns.flatten())

        logits, hidden = self.model(obs_next, state=None)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        q = logits[np.arange(len(logits)).tolist(), batch.act.tolist()]
        td_error = returns - q
        batch.weight = ops.stop_gradient(td_error)  # prio-buffer # todo: weights对梯度无贡献。

        import copy
        model = copy.deepcopy(self.model)
        optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-3)
        _clip_loss_grad = self._clip_loss_grad

        def forward_fn(act, returns, obs_next):
            logits, hidden = model(obs_next, state=None)
            q = logits[ms.numpy.arange(len(logits)), act]
            td_error = returns - q

            if _clip_loss_grad:
                y = q.reshape(-1, 1)
                t = returns.reshape(-1, 1)
                loss = nn.HuberLoss(reduction="mean")(y, t)
            else:
                loss = (td_error.pow(2) * weight).mean()

            return loss

        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
        @ms_function
        def train_step(obs, act, returns, obs_next):
            loss, grads = grad_fn(act, returns, obs_next)
            optimizer(grads)
            return loss,grads

        loss,grads = train_step(obs, act, returns, obs_next)
        self.model = copy.deepcopy(model)
        del model

        self._iter += 1
        print(loss.item((0)))
        return {"loss": loss.item((0))}

    def exploration_noise(
            self,
            act: Union[np.ndarray, Batch],
            batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act
