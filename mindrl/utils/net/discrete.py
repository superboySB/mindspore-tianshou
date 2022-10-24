from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Uniform, HeUniform, Constant, XavierUniform
from mindspore.nn.probability.distribution import Categorical
import math

from mindrl.data import Batch, to_mindspore
from mindrl.utils.net.common import MLP


class Actor(nn.Cell):
    """Simple actor network.

    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param bool softmax_output: whether to apply a softmax layer over the last
        layer's output.
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
            softmax_output: bool = True,
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
        self.softmax_output = softmax_output

    def construct(
            self,
            obs: Union[np.ndarray, ms.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[ms.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.preprocess(obs, state)
        logits = self.last(logits)
        if self.softmax_output:
            logits = nn.Softmax(axis=-1)(logits)
        return logits, hidden


class Critic(nn.Cell):
    """Simple critic network. Will create an actor operated in discrete \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int last_size: the output dimension of Critic network. Default to 1.
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
            hidden_sizes: Sequence[int] = (),
            last_size: int = 1,
            preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.output_dim = last_size
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            last_size,
            hidden_sizes,
        )

    def construct(
            self, obs: Union[np.ndarray, ms.Tensor], **kwargs: Any
    ) -> ms.Tensor:
        """Mapping: s -> V(s)."""
        logits, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(logits)


class CosineEmbeddingNetwork(nn.Cell):
    """Cosine embedding network for IQN. Convert a scalar in [0, 1] to a list \
    of n-dim vectors.

    :param num_cosines: the number of cosines used for the embedding.
    :param embedding_dim: the dimension of the embedding/output.

    .. note::

        From https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_cosines: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(num_cosines, embedding_dim, weight_init=HeUniform(negative_slope=math.sqrt(5)),
                     bias_init=Uniform(scale=1 / math.sqrt(num_cosines))), nn.ReLU())
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def construct(self, taus: ms.Tensor) -> ms.Tensor:
        batch_size = taus.shape[0]
        N = taus.shape[1]
        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * ms.numpy.arange(
            start=1, stop=self.num_cosines + 1, dtype=taus.dtype
        ).view(1, 1, self.num_cosines)
        # Calculate cos(i * \pi * \tau).
        cosines = ops.cos(taus.view(batch_size, N, 1) * i_pi
                          ).view(batch_size * N, self.num_cosines)
        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(batch_size, N, self.embedding_dim)
        return tau_embeddings


class ImplicitQuantileNetwork(Critic):
    """Implicit Quantile Network.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param int action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        Although this class inherits Critic, it is actually a quantile Q-Network
        with output shape (batch_size, action_dim, sample_size).

        The second item of the first return value is tau vector.
    """

    def __init__(
            self,
            preprocess_net: nn.Cell,
            action_shape: Sequence[int],
            hidden_sizes: Sequence[int] = (),
            num_cosines: int = 64,
            preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        last_size = int(np.prod(action_shape))
        super().__init__(
            preprocess_net, hidden_sizes, last_size, preprocess_net_output_dim
        )
        self.input_dim = getattr(
            preprocess_net, "output_dim", preprocess_net_output_dim
        )
        self.embed_model = CosineEmbeddingNetwork(
            num_cosines,
            self.input_dim  # type: ignore
        )

    def construct(  # type: ignore
            self, obs: Union[np.ndarray, ms.Tensor], sample_size: int, **kwargs: Any
    ) -> Tuple[Any, ms.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.preprocess(obs, state=kwargs.get("state", None))
        # Sample fractions.
        batch_size = logits.size(0)
        taus = ops.UniformReal()(
            (batch_size, sample_size)
        ).astype(dtype=logits.dtype)
        embedding = (ops.expand_dims(logits, 1) *
                     self.embed_model(taus)).view(batch_size * sample_size, -1)
        out = self.last(embedding).view(batch_size, sample_size, -1).transpose(0, 2, 1)
        return (out, taus), hidden


class FractionProposalNetwork(nn.Cell):
    """Fraction proposal network for FQF.

    :param num_fractions: the number of factions to propose.
    :param embedding_dim: the dimension of the embedding/input.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_fractions: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Dense(embedding_dim, num_fractions, weight_init=XavierUniform(gain=0.01),
                            bias_init=Constant(value=0))
        self.num_fractions = num_fractions
        self.embedding_dim = embedding_dim

    def construct(
            self, obs_embeddings: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        # Calculate (log of) probabilities q_i in the paper.
        dist = Categorical(probs=ops.Softmax(axis=-1)(self.net(obs_embeddings)))

        taus_1_N = ops.cumsum(dist.probs, axis=1)
        # Calculate \tau_i (i=0,...,N).
        taus = ops.pad(taus_1_N, paddings=((0, 0), (1, 0)))
        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1] + taus[:, 1:]) / 2.0
        # Calculate entropies of value distributions.
        entropies = dist.entropy()
        return taus, tau_hats, entropies


class FullQuantileFunction(ImplicitQuantileNetwork):
    """Full(y parameterized) Quantile Function.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param int action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        The first return value is a tuple of (quantiles, fractions, quantiles_tau),
        where fractions is a Batch(taus, tau_hats, entropies).
    """

    def __init__(
            self,
            preprocess_net: nn.Cell,
            action_shape: Sequence[int],
            hidden_sizes: Sequence[int] = (),
            num_cosines: int = 64,
            preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__(
            preprocess_net, action_shape, hidden_sizes, num_cosines,
            preprocess_net_output_dim
        )

    def _compute_quantiles(
            self, obs: ms.Tensor, taus: ms.Tensor
    ) -> ms.Tensor:
        batch_size, sample_size = taus.shape
        embedding = (ops.expand_dims(obs, axis=1) *
                     self.embed_model(taus)).view(batch_size * sample_size, -1)
        quantiles = self.last(embedding).view(batch_size, sample_size,
                                              -1).transpose(0, 2, 1)
        return quantiles

    def construct(  # type: ignore
            self, obs: Union[np.ndarray, ms.Tensor],
            propose_model: FractionProposalNetwork,
            fractions: Optional[Batch] = None,
            **kwargs: Any
    ) -> Tuple[Any, ms.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.preprocess(obs, state=kwargs.get("state", None))
        # Propose fractions
        if fractions is None:
            taus, tau_hats, entropies = propose_model(logits)
            fractions = Batch(taus=taus, tau_hats=tau_hats, entropies=entropies)
        else:
            taus, tau_hats = fractions.taus, fractions.tau_hats
        quantiles = self._compute_quantiles(logits, tau_hats)
        # Calculate quantiles_tau for computing fraction grad
        quantiles_tau = None
        if self.training:
            quantiles_tau = self._compute_quantiles(logits, taus[:, 1:-1])
        return (quantiles, fractions, quantiles_tau), hidden


class NoisyLinear(nn.Cell):
    """Implementation of Noisy Networks. arXiv:1706.10295.

    :param int in_features: the number of input features.
    :param int out_features: the number of output features.
    :param float noisy_std: initial standard deviation of noisy linear layers.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(
            self, in_features: int, out_features: int, noisy_std: float = 0.5
    ) -> None:
        super().__init__()


        # Learnable parameters.
        self.mu_W = ms.Parameter(ms.Tensor(np.ones((out_features, in_features)), ms.float32))
        self.sigma_W = ms.Parameter(ms.Tensor(np.ones((out_features, in_features)), ms.float32))
        self.mu_bias = ms.Parameter(ms.Tensor(np.ones((out_features,)), ms.float32))
        self.sigma_bias = ms.Parameter(ms.Tensor(np.ones((out_features,)), ms.float32))

        # Factorized noise parameters.
        self.register_buffer('eps_p', ms.Tensor(np.ones((in_features,)), ms.float32))
        self.register_buffer('eps_q', ms.Tensor(np.ones((out_features,)), ms.float32))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = noisy_std

        self.reset()
        self.sample()

    def reset(self) -> None:
        bound = 1 / np.sqrt(self.in_features)



        self.mu_W.set_data(Uniform(scale=bound))
        self.mu_bias.set_data(Uniform(scale=bound))
        self.sigma_W.set_data(Constant(self.sigma / np.sqrt(self.in_features)))
        self.sigma_bias.set_data(Constant(self.sigma / np.sqrt(self.in_features)))

    def f(self, x: ms.Tensor) -> ms.Tensor:
        x = ops.standard_normal(x.size(0))
        return ops.mul(ops.Sign()(x),ops.sqrt(x.abs()))

    def sample(self) -> None:
        self.eps_p = self.f(self.eps_p).copy()  # type: ignore
        self.eps_q = self.f(self.eps_q).copy()  # type: ignore

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.training:
            weight = self.mu_W + self.sigma_W * (
                self.eps_q.ger(self.eps_p)  # type: ignore
            )
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()  # type: ignore
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return ops.matmul(x,weight)+bias


def sample_noise(model: nn.Cell) -> bool:
    """Sample the random noises of NoisyLinear modules in the model.

    :param model: a PyTorch module which may have NoisyLinear submodules.
    :returns: True if model has at least one NoisyLinear submodule;
        otherwise, False.
    """
    done = False
    for m in model.cells():
        if isinstance(m, NoisyLinear):
            m.sample()
            done = True
    return done


class IntrinsicCuriosityModule(nn.Cell):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    :param torch.nn.Module feature_net: a self-defined feature_net which output a
        flattened hidden state.
    :param int feature_dim: input dimension of the feature net.
    :param int action_dim: dimension of the action space.
    :param hidden_sizes: hidden layer sizes for forward and inverse models.
    :param device: device for the module.
    """

    def __init__(
            self,
            feature_net: nn.Cell,
            feature_dim: int,
            action_dim: int,
            hidden_sizes: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.feature_net = feature_net
        self.forward_model = MLP(
            feature_dim + action_dim,
            output_dim=feature_dim,
            hidden_sizes=hidden_sizes,
        )
        self.inverse_model = MLP(
            feature_dim * 2,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )
        self.feature_dim = feature_dim
        self.action_dim = action_dim

    def construct(
            self, s1: Union[np.ndarray, ms.Tensor],
            act: Union[np.ndarray, ms.Tensor], s2: Union[np.ndarray,
                                                            ms.Tensor], **kwargs: Any
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        r"""Mapping: s1, act, s2 -> mse_loss, act_hat."""
        s1 = to_mindspore(s1, dtype=ms.float32)
        s2 = to_mindspore(s2, dtype=ms.float32)
        phi1, phi2 = self.feature_net(s1), self.feature_net(s2)
        act = to_mindspore(act, dtype=ms.int32, device=self.device)
        phi2_hat = self.forward_model(
            ops.concat([phi1,
                        ops.one_hot(indices=act,
                                    depth=self.action_dim,
                                    on_value=ms.Tensor(1.0, ms.float32),
                                    off_value=ms.Tensor(0, ms.float32))
                        ],
                       axis=1)
        )

        mse_loss = 0.5 * nn.MSELoss(reduction="none")(phi2_hat, phi2).sum(1)
        act_hat = self.inverse_model(ops.concat([phi1, phi2], axis=1))
        return mse_loss, act_hat
