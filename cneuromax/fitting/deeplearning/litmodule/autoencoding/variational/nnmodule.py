""":class:`BaseAutoEncoder` & its config."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated as An

from jaxtyping import Float
from omegaconf import MISSING
from torch import Tensor, nn

from cneuromax.utils.beartype import ge, lt


class BaseAutoEncoder(nn.Module, ABC):
    """AutoEncoder (MLP)."""

    @abstractmethod
    def encode(
        self: "BaseAutoEncoder",
        x: Float[Tensor, " batch_size *data_dim"],
    ) -> Float[Tensor, " batch_size *latent_dim"]:
        """Encodes the input data.

        Args:
            x: The input data batch.

        Returns:
            The latent representation.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(
        self: "BaseAutoEncoder",
        x: Float[Tensor, " batch_size *latent_dim"],
    ) -> Float[Tensor, " batch_size *data_dim"]:
        """Decodes the latent representation.

        Args:
            x: The latent representation.

        Returns:
            The reconstructed input data.
        """
        raise NotImplementedError

    def forward(
        self: "BaseAutoEncoder",
        x: Float[Tensor, " batch_size *data_dim"],
    ) -> Float[Tensor, " batch_size *data_dim"]:
        """.

        Args:
            x: The input data batch.

        Returns:
            The output batch.
        """
        latent: Float[Tensor, " batch_size *latent_dim"] = self.encode(x)
        return self.decode(latent)
