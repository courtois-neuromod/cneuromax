import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from typeguard import typechecked

from cntrain.dl.model.unimodal.common.autoencoding.ae import BaseAEModel


class BaseVAEModel(BaseAEModel):
    """Base Variational Autoencoder Model."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        """Constructor.

        Args:
            optimizer: Optimizer.
        """

        super().__init__(optimizer)

    @typechecked
    def forward(
        self, x: Float[torch.Tensor, "batch_size *x_shape"]
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

    @typechecked
    def step(
        self,
        batch: Float[torch.Tensor, "batch_size *x_shape"],
        stage: str,
    ):

        x = batch

        # BS x (XS) -> BS x (XS), BS x (LS), BS x (LS)
        recon_x, mu, log_sigma = self(x)
        recon_loss, kl_loss = self.compute_losses(x, recon_x, mu, log_sigma)

        self.log(f"{stage}/recon_loss", recon_loss)
        self.log(f"{stage}/kl_loss", kl_loss)

        return recon_loss + cfg.beta * kl_loss

    def compute_losses(self, x, recon_x, mu, log_sigma):
        """
        Args:
            x: batch_size x (x_shape)
            recon_x: batch_size x (x_shape)
            mu: batch_size x (latent_shape)
            log_sigma: batch_size x (latent_shape)
        Returns:
            recon_loss: 1
            kl_loss: 1
        """
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        kl_loss = -0.5 * torch.sum(
            1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
        )

        return recon_loss, kl_loss
