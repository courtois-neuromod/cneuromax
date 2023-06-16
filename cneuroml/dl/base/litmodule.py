"""Base LitModule.

Abbreviations used in this module:

Lightning's ``LightningModule`` is short for
``lightning.pytorch.LightningModule``.

PyTorch ``nn.Module`` is short for ``torch.nn.Module``.

PyTorch ``Optimizer`` is short for ``torch.optim.Optimizer``.

PyTorch ``LRScheduler`` is short for
``torch.optim.lr_scheduler.LRScheduler``.

``Float`` is short for ``jaxtyping.Float``.
"""

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
    """The Base LitModule class.

    This class inherits from Lightning's ``LightningModule`` class.
    Subclasses need to implement the ``step`` instance method.

    Attributes:
        nnmodule (``nn.Module``): The PyTorch ``nn.Module`` instance.
        optimizer (``Optimizer``): The PyTorch ``Optimizer`` instance.
        scheduler (``LRScheduler``): The PyTorch ``LRScheduler``
            instance.
    """

    def __init__(
        self: "BaseLitModule",
        nnmodule: nn.Module,
        optimizer_partial: partial[Optimizer],
        scheduler_partial: partial[LRScheduler],
    ) -> None:
        """Constructor.

        Calls parent constructor, stores the ``nnmodule`` and
        instantiates the ``optimizer`` and ``scheduler`` instance
        attributes.

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
        """Training step method.

        Args:
            batch: An input data batch (images/sound/language/...).

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
        """Validation step method.

        Args:
            batch: An input data batch (images/sound/language/...).

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
        """Tests step method.

        Args:
            batch: An input data batch (images/sound/language/...).

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
        """Returns ``Optimizer`` & ``LRScheduler`` instance attributes.

        Returns:
            A tuple containing the PyTorch ``Optimizer`` and
            ``LRScheduler`` instance attributes (each nested in a
            list)
        """
        return [self.optimizer], [self.scheduler]
