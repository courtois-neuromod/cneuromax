""":class:`BaseAutoencodingLitModule` & its config."""

import torch
import torch.nn.functional as f
from jaxtyping import Float
from torch import nn
from typeguard import typechecked

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule


class BaseAutoencodingLitModule(BaseLitModule):
    """Autoencoder model."""

    def __init__(
        self: "BaseAutoencodingLitModule",
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
        self: "BaseAutoencodingLitModule",
        x: Float[torch.Tensor, "batch_size *x_shape"],
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
        self: "BaseAutoencodingLitModule",
        batch: Float[torch.Tensor, " batch_size *x_shape"],
    ) -> Float[torch.Tensor, ""]:
        """
        Args:
            batch: The input batch.
        Returns:
            The loss (reconstruction loss).
        """
        x = batch
        recon_x = self(x)
        recon_loss = f.mse_loss(x, recon_x)

        return recon_loss

    @typechecked
    def encode(
        self: "BaseAutoencodingLitModule",
        x: Float[torch.Tensor, "batch_size *x_shape"],
    ) -> Float[torch.Tensor, "batch_size *latent_shape"]:

        return self.encoder(x)

    def decode(
        self: "BaseAutoencodingLitModule",
        x: Float[torch.Tensor, " batch_size *latent_shape"],
    ) -> Float[torch.Tensor, " batch_size *x_shape"]:

        return self.decoder(x)
