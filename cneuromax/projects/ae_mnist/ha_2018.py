from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from ...fitting.deeplearning.litmodule.autoencoding.nnmodule import (
    BaseAutoEncoder,
)


@dataclass
class Ha2018AutoencoderConfig:
    """Holds :class:`Ha2018Autoencoder` config values.

    Args:
        num_channels: Number of input channels.
        latent_size: Size of the latent representation vector.
    """

    num_channels: int = 3
    latent_size: int = 32


class Ha2018Autoencoder(BaseAutoEncoder):
    """Autoencoder from "World Models" (Ha & Schmidhuber, 2018).

    Paper link: https://arxiv.org/abs/1803.10122

    Args:
        config: See :class:`Ha2018AutoencoderConfig`.

    Attributes:
        config: See :paramref:`config`.
    """

    def __init__(
        self: "Ha2018Autoencoder",
        config: Ha2018AutoencoderConfig,
    ) -> None:
        self.config = config
        self.encoder = Ha2018Encoder(
            num_channels=config.num_channels,
            latent_size=config.latent_size,
        )
        self.decoder = Ha2018Decoder(
            num_channels=config.num_channels,
            latent_size=config.latent_size,
        )

    def encode(
        self: "BaseAutoEncoder",
        x: Float[Tensor, " batch_size num_channels "],
    ) -> Float[Tensor, " batch_size *latent_dim"]:
        """Encodes input data :paramref:`x`.

        Args:
            x: The input data batch.

        Returns:
            The latent representation.
        """


class MNISTConvEncoder(nn.Module):

    def __init__(self: "MNISTEncoder") -> None:
        # BS x NC x 64 x 64 -> BS x 32 x 31 x 31
        self.conv1 = nn.Conv2d(
            in_channels=self.config.num_channels,
            out_channels=32,
            kernel_size=4,
            stride=2,
        )
        # BS x 32 x 31 x 31 -> BS x 64 x 14 x 14
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
        )
        # BS x 64 x 14 x 14 -> BS x 128 x 6 x 6
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
        )
        # BS x 128 x 6 x 6 -> BS x 256 x 2 x 2
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=4,
            stride=2,
        )
        # BS x 1024 -> BS x LS
        self.fc1 = nn.Linear(
            in_features=256 * 2 * 2,
            out_features=self.config.latent_size,
        )


class Ha2018Decoder(nn.Module):
    def initialize_decoder_layers(self: "Ha2018Autoencoder") -> None:
        """Self-explanatory."""
        # BS x LS -> BS x 1024
        self.fc2 = nn.Linear(
            in_features=self.config.latent_size,
            out_features=256 * 2 * 2,
        )
        # BS x 1024 x 1 x 1 -> BS x 128 x 5 x 5
        self.tconv1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=128,
            kernel_size=5,
            stride=2,
        )
        # BS x 128 x 5 x 5 -> BS x 64 x 13 x 13
        self.tconv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=5,
            stride=2,
        )
        # BS x 64 x 13 x 13 -> BS x 32 x 30 x 30
        self.tconv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=6,
            stride=2,
        )
        # BS x 32 x 30 x 30 -> BS x 3 x 64 x 64
        self.tconv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=self.config.num_channels,
            kernel_size=6,
            stride=2,
        )
