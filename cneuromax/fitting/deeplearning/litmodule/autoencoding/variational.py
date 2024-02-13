""":class:`BaseVariationalAutoencodingLitModule` & its config."""

from dataclasses import dataclass
from typing import Annotated as An

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as f
from jaxtyping import Float
from torch import Tensor
from typeguard import typechecked

from cneuromax.utils.beartype import one_of

from .base import BaseAutoencodingLitModule


@dataclass
class BaseVAELitModuleConfig:
    """Holds :class:`BaseVAELitModule` config values.

    Args:
        beta: The scaling factor for the KL divergence term, see\
            https://openreview.net/forum?id=Sy2fzU9gl.
    """

    beta: float = 1.0


class BaseVAELitModule(BaseAutoencodingLitModule):
    """Base Variational Autoencoder Model."""

    def __init__(
        self: "BaseVAELitModule",
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(*args, **kwargs)

    @typechecked
    def forward(
        self: "BaseVAELitModule",
        x: Float[torch.Tensor, " batch_size *x_shape"],
    ) -> tuple(
        Float[torch.Tensor, "batch_size *x_shape"],
        Float[torch.Tensor, "batch_size *latent_shape"],
        Float[torch.Tensor, "batch_size *latent_shape"],
    ):
        """Forward method.
        Args:
            x: Input features/sounds/images...
        Returns:
            The reconstructed features/sounds/images...
            The gaussian means (mu).
            The gaussian log standard deviations (log_sigma).
        """
        # BS x (LS*2) -> BS x (LS), BS x (LS)
        mu, log_sigma = self.encoder(x).split(self.latent_shape[0], dim=1)

        sigma = torch.exp(log_sigma)

        epsilon = torch.randn(self.latent_shape, device=self.device)

        z = mu + sigma * epsilon

        # BS x (LS) -> BS x (XS)
        recon_x = self.decoder(z)

        return recon_x, mu, log_sigma

    def step(
        self: "BaseVAELitModule",
        batch: Float[Tensor, " batch_size *data_dim"],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, ""]:
        """Computes the mean squared error loss for the autoencoder.

        Args:
            batch: The input data batch.
            stage: See\
                :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The mean squared error loss.
        """
        x = batch
        output = self.nnmodule(x)
        recon_x: Float[Tensor, " batch_size *data_dim"] = output[0]
        mu: Float[Tensor, " batch_size *latent_dim"] = output[1]
        log_sigma: Float[Tensor, " batch_size *latent_dim"] = output[2]
        recon_loss = F.mse_loss(input=recon_x, target=x, reduction="sum")
        kl_loss = -0.5 * torch.sum(
            input=1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp(),
        )
        self.log(f"{stage}/recon_loss", recon_loss)
        self.log(f"{stage}/kl_loss", kl_loss)

        return recon_loss + cfg.beta * kl_loss
