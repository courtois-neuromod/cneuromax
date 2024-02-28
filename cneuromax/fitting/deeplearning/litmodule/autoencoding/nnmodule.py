""":class:`BaseAutoEncoder`."""

from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor, nn


class BaseAutoEncoder(nn.Module, ABC):
    """Base AutoEncoder :class:`torch.nn.Module`."""

    @abstractmethod
    def encode(
        self: "BaseAutoEncoder",
        x: Float[Tensor, " batch_size *data_dim"],
    ) -> Float[Tensor, " batch_size *latent_dim"]:
        """Encodes input data :paramref:`x`.

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
        """Decodes latent representation :paramref:`x`.

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
        """Encodes then decodes input data :paramref:`x`.

        Args:
            x: The input data batch.

        Returns:
            The output batch.
        """
        latent: Float[Tensor, " batch_size *latent_dim"] = self.encode(x)
        return self.decode(latent)
