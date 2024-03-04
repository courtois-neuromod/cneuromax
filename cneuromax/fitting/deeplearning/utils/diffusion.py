"""Diffusion models."""

import math

import torch
import torch.nn.functional as f
from jaxtyping import Float, Int
from torch import Tensor, nn


class GaussianDiffusion:
    """Gaussian diffusion model.

    Replication of (Ho et al., 2020) with the following modifications:
    - ``alpha_t`` & ``beta_t`` follow a cosine schedule as in
    (Nichol & Dhariwal, 2021).

    Papers:
    - Denoising Diffusion Probabilistic Models
    (Ho et al., 2020) https://arxiv.org/abs/2006.11239
    - Improved Denoising Diffusion Probabilistic Models
    (Nichol & Dhariwal, 2021) https://arxiv.org/abs/2102.09672

    Args:
        nnmodule: The PyTorch network module.
        big_t: The total number of diffusion timesteps (``T`` in paper).

    Attributes:
        nnmodule: See :paramref:`nnmodule`.
        big_t: See :paramref:`big_t`.
        all_alpha_t_bar: ``alpha_t_bar`` for all diffusion timesteps.
        all_beta_t: ``beta_t`` for all diffusion timesteps.
    """

    def __init__(
        self: "GaussianDiffusion",
        nnmodule: nn.Module,
        big_t: int,
    ) -> None:
        self.nnmodule = nnmodule
        self.big_t = big_t
        self.compute_all_alpha_t_bar()
        self.compute_all_beta_t()

    def compute_all_alpha_t_bar(self: "GaussianDiffusion") -> None:
        """Equation 17 from Nichol & Dhariwal (2021).

        Paper name: Improved Denoising Diffusion Probabilistic Models
        Link: https://arxiv.org/abs/2102.09672, page 4

        Computes :attr:`all_alpha_t_bar`.
        """
        all_t: Float[Tensor, " T_plus_1"] = torch.linspace(
            start=0,
            end=self.big_t,
            steps=self.big_t + 1,
        )
        s = 0.008
        all_f_t: Float[Tensor, " T_plus_1"] = torch.pow(
            input=torch.cos(
                input=(all_t / self.big_t + s) / (1 + s) * math.pi / 2,
            ),
            exponent=2,
        )
        self.all_alpha_t_bar: Float[Tensor, " T_plus_1"] = all_f_t / all_f_t[0]

    def compute_all_beta_t(self: "GaussianDiffusion") -> None:
        """After equation 17 in Nichol & Dhariwal (2021).

        Paper name: Improved Denoising Diffusion Probabilistic Models
        Link: https://arxiv.org/abs/2102.09672, page 4
        """
        all_beta_t: Float[Tensor, " T"] = 1 - (
            self.all_alpha_t_bar[1:] / self.all_alpha_t_bar[:-1]
        )
        self.all_beta_t: Float[Tensor, " T"] = torch.clip(
            input=all_beta_t,
            min=0,
            max=0.999,
        )

    def sample_from_q_x_t_given_x_0(
        self: "GaussianDiffusion",
        x_0: Float[Tensor, " batch_size *x_dim"],
        t: Int[Tensor, " batch_size"],
    ) -> tuple[
        Float[Tensor, " batch_size *x_dim"],
        Float[Tensor, " batch_size *x_dim"],
    ]:
        """Equation 4 in Ho et al. (2020).

        Args:
            x_0: Batched data from the data distribution.
            t: Batched timesteps up to which to noise the data.

        Returns:
            Batched samples from the distribution :math:`q(x_t|x_0)` &\
                batched ``epsilon``.
        """
        epsilon: Float[Tensor, " batch_size *x_dim"] = torch.randn_like(
            input=x_0,
        )
        q_x_t_given_x_0_sample: Float[Tensor, " batch_size *x_dim"] = (
            self.all_alpha_t_bar[t].sqrt() * x_0
            + (1 - self.all_alpha_t_bar[t]).sqrt() * epsilon
        )
        return q_x_t_given_x_0_sample, epsilon

    def sample_from_p_x_t_given_x_t_plus_1(
        self: "GaussianDiffusion",
        x_t_plus_1: Float[Tensor, " batch_size *x_dim"],
        t: Int[Tensor, " batch_size"],
    ) -> tuple[
        Float[Tensor, " batch_size *x_dim"],
        Float[Tensor, " batch_size *x_dim"],
    ]:
        """Equation 5 in Ho et al. (2020).

        Args:
            x_t_plus_1: Batched data from the data distribution.
            t: Batched timesteps up to which to noise the data.

        Returns:
            Batched samples from the distribution :math:`p(x_t|x_{t+1})` &\
                batched ``epsilon``.
        """
        epsilon: Float[Tensor, " batch_size *x_dim"] = torch.randn_like(
            input=x_t_plus_1,
        )
        p_x_t_given_x_t_plus_1_sample: Float[Tensor, " batch_size *x_dim"] = (
            self.all_beta_t[t].sqrt() * x_t_plus_1
            + (1 - self.all_beta_t[t]).sqrt() * epsilon
        )
        return p_x_t_given_x_t_plus_1_sample, epsilon

    def compute_l_simple(
        self: "GaussianDiffusion",
        x_0: Float[Tensor, " batch_size *x_dim"],
    ) -> Float[Tensor, " "]:
        """Equation 14 in Ho et al. (2020).

        Paper name: Improved Denoising Diffusion Probabilistic Models
        Link: https://arxiv.org/abs/2102.09672, page 3

        Args:
            x_0: Batched data from the data distribution.
        """
        t: Int[Tensor, " batch_size"] = torch.randint(
            low=0,
            high=self.big_t,
            size=(x_0.shape[0],),
            dtype=torch.int,
        )
        out: tuple[
            Float[Tensor, " batch_size *x_dim"],
            Float[Tensor, " batch_size *x_dim"],
        ] = self.sample_from_q_x_t_given_x_0(
            x_0=x_0,
            t=t,
        )
        q_x_t_given_x_0_sample: Float[Tensor, " batch_size *x_dim"] = out[0]
        epsilon: Float[Tensor, " batch_size *x_dim"] = out[1]
        epsilon_theta: Float[Tensor, " batch_size *x_dim"] = self.nnmodule(
            x=q_x_t_given_x_0_sample,
            t=t,
        )
        return f.mse_loss(input=epsilon_theta, target=epsilon)
