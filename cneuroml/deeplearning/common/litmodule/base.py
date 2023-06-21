"""Base LitModule."""

from abc import ABCMeta
from functools import partial
from typing import final

from beartype import beartype as typechecker
from jaxtyping import Float
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class BaseLitModule(LightningModule, metaclass=ABCMeta):
    """.

    Subclasses need to implement the ``step`` instance method.

    Attributes:
        nnmodule (``nn.Module``): .
        optimizer (``Optimizer``): .
        scheduler (``LRScheduler``): .
    """

    def __init__(
        self: "BaseLitModule",
        nnmodule: nn.Module,
        optimizer_partial: partial[Optimizer],
        scheduler_partial: partial[LRScheduler],
    ) -> None:
        """.

        Calls parent constructor, stores arguments, and instantiates
        from partial functions.

        Args:
            nnmodule: A PyTorch ``nn.Module`` instance.
            optimizer_partial: A PyTorch ``Optimizer`` partial function.
            scheduler_partial: A PyTorch ``LRScheduler`` partial
                function.
        """
        super().__init__()

        self.nnmodule = nnmodule
        self.optimizer = optimizer_partial(params=self.parameters())
        self.scheduler = scheduler_partial(optimizer=self.optimizer)

    @final
    @typechecker
    def training_step(
        self: "BaseLitModule",
        batch: Float[Tensor, "..."] | tuple[Float[Tensor, "..."], ...],
    ) -> Float[Tensor, "..."]:
        """.

        Args:
            batch: .

        Returns:
            The loss value(s).

        Raises:
            AttributeError: If the ``step`` instance method is not
                callable.
        """
        if not (hasattr(self, "step") and callable(self.step)):
            raise AttributeError

        loss = self.step(batch, "train")
        self.log("train/loss", loss)

        return loss

    @final
    @typechecker
    def validation_step(
        self: "BaseLitModule",
        batch: Float[Tensor, "..."] | tuple[Float[Tensor, ""], ...],
    ) -> Float[Tensor, ""]:
        """.

        Args:
            batch: .

        Returns:
            The loss value(s).

        Raises:
            AttributeError: If the ``step`` instance method is not
                callable.
        """
        if not (hasattr(self, "step") and callable(self.step)):
            raise AttributeError

        loss = self.step(batch, "val")
        self.log("val/loss", loss)

        return loss

    @final
    @typechecker
    def test_step(
        self: "BaseLitModule",
        batch: Float[Tensor, "..."] | tuple[Float[Tensor, ""], ...],
    ) -> Float[Tensor, ""]:
        """.

        Args:
            batch: .

        Returns:
            The loss value(s).

        Raises:
            AttributeError: If ``step`` instance method is not
                callable.
        """
        if not (hasattr(self, "step") and callable(self.step)):
            raise AttributeError

        loss = self.step(batch, "test")
        self.log("test/loss", loss)

        return loss

    @final
    def configure_optimizers(
        self: "BaseLitModule",
    ) -> tuple[list[Optimizer], list[LRScheduler]]:
        """.

        Returns:
            A tuple containing the PyTorch ``Optimizer`` and
            ``LRScheduler`` instance attributes (each nested in a
            list).
        """
        return [self.optimizer], [self.scheduler]
