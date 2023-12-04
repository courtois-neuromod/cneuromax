import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Shaped
from typeguard import typechecked

from cntrain.dl.common.base.model import BaseModel


class AEModel(BaseModel):
    """Autoencoder model."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        """Constructor.

        Args:
            encoder: Encoder model.
            decoder: Decoder model.
        """
        super().__init__(optimizer)

        self.encoder, self.decoder = encoder, decoder

    @typechecked
    def forward(
        self, x: Float[torch.Tensor, "batch_size *x_shape"]
    ) -> Float[torch.Tensor, "batch_size *x_shape"]:
        """Forward method.

        Args:
            x: The input batch.
        Returns:
            The reconstructed input batch.
        """
        latent = self.encoder(x)
        recon_x = self.decoder(latent)

        return recon_x

    @typechecked
    def step(
        self, batch: Float[torch.Tensor, "batch_size *x_shape"]
    ) -> Float[torch.Tensor, ""]:
        """
        Args:
            batch: The input batch.
        Returns:
            The loss (reconstruction loss).
        """
        x = batch
        recon_x = self(x)
        recon_loss = F.mse_loss(x, recon_x)

        return recon_loss

    @typechecked
    def encode(
        x: Float[torch.Tensor, "batch_size *x_shape"]
    ) -> Float[torch.Tensor, "batch_size *latent_shape"]:

        return self.encoder(x)

    @typechecked
    def decode(
        x: Float[torch.Tensor, "batch_size *latent_shape"]
    ) -> Float[torch.Tensor, "batch_size *x_shape"]:

        return self.decoder(x)
