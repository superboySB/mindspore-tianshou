from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np

from mindrl.utils.net.common import MLP


import mindspore as ms
from mindspore import nn,ops

SIGMA_MIN = -20
SIGMA_MAX = 2


class Actor(nn.Cell):
    """Simple actor network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~mindrl.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Cell,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
        )
        self._max = max_action

    def construct(
        self,
        obs: Union[np.ndarray, ms.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[ms.Tensor, Any]:
        """Mapping: obs -> logits -> action."""
        logits, hidden = self.preprocess(obs, state)
        logits = self._max * ops.tanh(self.last(logits))
        return logits, hidden


class Critic(nn.Cell):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~mindrl.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Cell,
        hidden_sizes: Sequence[int] = (),
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Dense] = nn.Dense,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            1,
            hidden_sizes,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )

    def construct(
        self,
        obs: Union[np.ndarray, ms.Tensor],
        act: Optional[Union[np.ndarray, ms.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> ms.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs = ms.Tensor(
            obs,
            dtype=ms.float32,
        )
        obs = ops.flatten(obs)
        if act is not None:
            act = ms.Tensor(
                act,
                dtype=ms.float32,
            )
            act = ops.flatten(act)
            obs = ops.concat([obs, act], axis=1)
        logits, hidden = self.preprocess(obs)
        logits = self.last(logits)
        return logits


class ActorProb(nn.Cell):
    """Simple actor network (output with a Gauss distribution).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param bool unbounded: whether to apply tanh activation on final logits.
        Default to False.
    :param bool conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter. Default to False.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~mindrl.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Cell,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.mu = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
        )
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim,  # type: ignore
                self.output_dim,
                hidden_sizes,
            )
        else:
            self.sigma_param = ms.Parameter(ops.zeros((self.output_dim, 1),ms.float32))
        self._max = max_action
        self._unbounded = unbounded

    def forward(
        self,
        obs: Union[np.ndarray, ms.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[ms.Tensor, ms.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * ops.tanh(mu)
        if self._c_sigma:
            sigma = ops.clip_by_value(
                self.sigma(logits),
                ms.Tensor(SIGMA_MIN,ms.float32),
                ms.Tensor(SIGMA_MAX,ms.float32)
            )
            sigma = ops.exp(sigma)
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + ops.zeros_like(mu))
            sigma = ops.exp(sigma)
        return (mu, sigma), state


class RecurrentActorProb(nn.Cell):
    """Recurrent version of ActorProb.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        hidden_layer_size: int = 128,
        max_action: float = 1.0,
        unbounded: bool = False,
        conditioned_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        output_dim = int(np.prod(action_shape))
        self.mu = nn.Dense(hidden_layer_size, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Dense(hidden_layer_size, output_dim)
        else:
            self.sigma_param = ms.Parameter(ops.zeros((output_dim, 1),ms.float32))
        self._max = max_action
        self._unbounded = unbounded

    def forward(
        self,
        obs: Union[np.ndarray, ms.Tensor],
        state: Optional[Dict[str, ms.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[ms.Tensor, ms.Tensor], Dict[str, ms.Tensor]]:
        """Almost the same as :class:`~mindrl.utils.net.common.Recurrent`."""
        obs = ms.Tensor(
            obs,
            dtype=ms.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(obs.shape) == 2:
            obs = ops.expand_dims(obs,axis=-2)

        # self.nn.flatten_parameters()

        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, (hidden, cell) = self.nn(
                obs, (
                    state["hidden"].transpose((1,0,2)),
                    state["cell"].transpose((1,0,2))
                )
            )
        logits = obs[:, -1]
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * ops.tanh(mu)
        if self._c_sigma:
            sigma = ops.clip_by_value(
                self.sigma(logits),
                ms.Tensor(SIGMA_MIN,ms.float32),
                ms.Tensor(SIGMA_MAX,ms.float32)
            )
            sigma = ops.exp(sigma)
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + ops.zeros_like(mu))
            sigma = ops.exp(sigma)
        # please ensure the first dim is batch size: [bsz, len, ...]
        return (mu, sigma), {
            "hidden": hidden.transpose((1, 0, 2)),
            "cell": cell.transpose((1,0,2))
        }


class RecurrentCritic(nn.Cell):
    """Recurrent version of Critic.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int] = [0],
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc2 = nn.Dense(hidden_layer_size + int(np.prod(action_shape)), 1)

    def forward(
        self,
        obs: Union[np.ndarray, ms.Tensor],
        act: Optional[Union[np.ndarray, ms.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> ms.Tensor:
        """Almost the same as :class:`~mindrl.utils.net.common.Recurrent`."""
        obs = ms.Tensor(
            obs,
            dtype=ms.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        assert len(obs.shape) == 3
        self.nn.flatten_parameters()
        obs, (hidden, cell) = self.nn(obs)
        obs = obs[:, -1]
        if act is not None:
            act = ms.Tensor(
                act,
                dtype=ms.float32,
            )
            obs = ops.concat([obs, act], axis=1)
        obs = self.fc2(obs)
        return obs


class Perturbation(nn.Cell):
    """Implementation of perturbation network in BCQ algorithm. Given a state and \
    action, it can generate perturbed action.

    :param torch.nn.Module preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param float max_action: the maximum value of each dimension of action.
    :param Union[str, int, torch.device] device: which device to create this model on.
        Default to cpu.
    :param float phi: max perturbation parameter for BCQ. Default to 0.05.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        You can refer to `examples/offline/offline_bcq.py` to see how to use it.
    """

    def __init__(
        self,
        preprocess_net: nn.Cell,
        max_action: float,
        phi: float = 0.05
    ):
        # preprocess_net: input_dim=state_dim+action_dim, output_dim=action_dim
        super(Perturbation, self).__init__()
        self.preprocess_net = preprocess_net
        self.max_action = max_action
        self.phi = phi

    def construct(self, state: ms.Tensor, action: ms.Tensor) -> ms.Tensor:
        # preprocess_net
        logits = self.preprocess_net(ops.concat([state, action], -1))[0]
        noise = self.phi * self.max_action * ops.tanh(logits)
        # clip to [-max_action, max_action]
        return ops.clip_by_value(noise + action,ms.Tensor(-self.max_action), ms.Tensor(self.max_action))


class VAE(nn.Cell):
    """Implementation of VAE. It models the distribution of action. Given a \
    state, it can generate actions similar to those in batch. It is used \
    in BCQ algorithm.

    :param torch.nn.Module encoder: the encoder in VAE. Its input_dim must be
        state_dim + action_dim, and output_dim must be hidden_dim.
    :param torch.nn.Module decoder: the decoder in VAE. Its input_dim must be
        state_dim + latent_dim, and output_dim must be action_dim.
    :param int hidden_dim: the size of the last linear-layer in encoder.
    :param int latent_dim: the size of latent layer.
    :param float max_action: the maximum value of each dimension of action.
    :param Union[str, torch.device] device: which device to create this model on.
        Default to "cpu".

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        You can refer to `examples/offline/offline_bcq.py` to see how to use it.
    """

    def __init__(
        self,
        encoder: nn.Cell,
        decoder: nn.Cell,
        hidden_dim: int,
        latent_dim: int,
        max_action: float,
    ):
        super(VAE, self).__init__()
        self.encoder = encoder

        self.mean = nn.Dense(hidden_dim, latent_dim)
        self.log_std = nn.Dense(hidden_dim, latent_dim)

        self.decoder = decoder

        self.max_action = max_action
        self.latent_dim = latent_dim

    def construct(
        self, state: ms.Tensor, action: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        # [state, action] -> z , [state, z] -> action
        latent_z = self.encoder(ops.concat([state, action], -1))
        # shape of z: (state.shape[:-1], hidden_dim)

        mean = self.mean(latent_z)
        # Clamped for numerical stability
        log_std = ops.clip_by_value(self.log_std(latent_z),ms.Tensor(-4,ms.float32), ms.Tensor(15,ms.float32))
        std = ops.exp(log_std)
        # shape of mean, std: (state.shape[:-1], latent_dim)

        latent_z = mean + std * ops.standard_normal(std.shape)  # (state.shape[:-1], latent_dim)

        reconstruction = self.decode(state, latent_z)  # (state.shape[:-1], action_dim)
        return reconstruction, mean, std

    def decode(
        self,
        state: ms.Tensor,
        latent_z: Union[ms.Tensor, None] = None
    ) -> ms.Tensor:
        # decode(state) -> action
        if latent_z is None:
            # state.shape[0] may be batch_size
            # latent vector clipped to [-0.5, 0.5]
            latent_z = ops.standard_normal(state.shape[:-1] + (self.latent_dim, ))
            latent_z = ops.clip_by_value(latent_z, ms.Tensor(-0.5, ms.float32), ms.Tensor(0.5, ms.float32))
        # decode z with state!
        return self.max_action * \
            ops.tanh(self.decoder(ops.concat([state, latent_z], -1)))
