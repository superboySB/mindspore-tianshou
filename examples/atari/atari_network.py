from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import nn, ops

from mindrl.utils.net.discrete import NoisyLinear


class DQN(nn.Cell):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int],
            features_only: bool = False,
            output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.output_dim = int(np.prod(self.net(ops.zeros((1, c, h, w), ms.float32)).shape[1:]))
        if not features_only:
            self.net = nn.SequentialCell(
                self.net, nn.Dense(self.output_dim, 512), nn.ReLU(),
                nn.Dense(512, int(np.prod(action_shape)))
            )
            self.output_dim = int(np.prod(action_shape))
        elif output_dim is not None:
            self.net = nn.SequentialCell(
                self.net, nn.Dense(self.output_dim, output_dim),
                nn.ReLU()
            )
            self.output_dim = output_dim

    def construct(
            self,
            obs: Union[np.ndarray, ms.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[ms.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = ms.Tensor(obs, dtype=ms.float32)
        return self.net(obs), state


class C51(DQN):
    """Reference: A distributional perspective on reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int],
            num_atoms: int = 51,
    ) -> None:
        self.action_num = np.prod(action_shape)
        super().__init__(c, h, w, [self.action_num * num_atoms])
        self.num_atoms = num_atoms

    def construct(
            self,
            obs: Union[np.ndarray, ms.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[ms.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().construct(obs)
        obs = ops.softmax(obs.view(-1, self.num_atoms), axis=-1)
        obs = obs.view(-1, self.action_num, self.num_atoms)
        return obs, state


class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int],
            num_atoms: int = 51,
            noisy_std: float = 0.5,
            is_dueling: bool = True,
            is_noisy: bool = True,
    ) -> None:
        super().__init__(c, h, w, action_shape, features_only=True)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            else:
                return nn.Dense(x, y)

        self.Q = nn.SequentialCell(
            linear(self.output_dim, 512), nn.ReLU(),
            linear(512, self.action_num * self.num_atoms)
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.SequentialCell(
                linear(self.output_dim, 512), nn.ReLU(),
                linear(512, self.num_atoms)
            )
        self.output_dim = self.action_num * self.num_atoms

    def construct(
            self,
            obs: Union[np.ndarray, ms.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[ms.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().construct(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(axis=1, keepdim=True) + v
        else:
            logits = q
        probs = ops.softmax(logits, axis=2)
        return probs, state


class QRDQN(DQN):
    """Reference: Distributional Reinforcement Learning with Quantile \
    Regression.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int],
            num_quantiles: int = 200,
    ) -> None:
        self.action_num = np.prod(action_shape)
        super().__init__(c, h, w, [self.action_num * num_quantiles])
        self.num_quantiles = num_quantiles

    def construct(
            self,
            obs: Union[np.ndarray, ms.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[ms.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().construct(obs)
        obs = obs.view(-1, self.action_num, self.num_quantiles)
        return obs, state
