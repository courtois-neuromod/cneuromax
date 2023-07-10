"""."""

from abc import ABCMeta
from functools import partial
from typing import Any, final

from beartype import beartype as typechecker
from jaxtyping import Num
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
        lrscheduler (``LRScheduler``): .
    """

    def __init__(
        self: "BaseLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        lrscheduler: partial[LRScheduler],
    ) -> None:
        """.

        Calls parent constructor, stores arguments, and instantiates
        from partial functions.

        Args:
            nnmodule: A PyTorch ``nn.Module`` instance.
            optimizer: A PyTorch ``Optimizer`` partial function.
            lrscheduler: A PyTorch ``LRScheduler`` partial
                function.
        """
        super().__init__()

        self.nnmodule = nnmodule
        self.optimizer = optimizer(params=self.parameters())
        self.lrscheduler = lrscheduler(optimizer=self.optimizer)

    @final
    @typechecker
    def training_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
    ) -> Num[Tensor, " ..."]:
        """.

        Args:
            batch: .

        Returns:
            The loss value(s).

        Raises:
            AttributeError: If the ``step`` instance method is not
                callable.
        """
        if not hasattr(self, "step"):
            raise AttributeError

        if not (hasattr(self, "step") and callable(self.step)):
            raise AttributeError

        if isinstance(batch, list):
            batch = tuple(batch)

        loss = self.step(batch, "train")
        self.log("train/loss", loss)

        return loss

    @final
    @typechecker
    def validation_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Num[Tensor, " ..."]:
        """.

        Args:
            batch: .
            *args: .
            **kwargs: .

        Returns:
            The loss value(s).

        Raises:
            AttributeError: If the ``step`` instance method is not
                callable.
        """
        if not (hasattr(self, "step") and callable(self.step)):
            raise AttributeError

        if isinstance(batch, list):
            batch = tuple(batch)

        loss = self.step(batch, "val")
        self.log("val/loss", loss)

        return loss

    @final
    @typechecker
    def test_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
    ) -> Num[Tensor, " ..."]:
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

        if isinstance(batch, list):
            batch = tuple(batch)

        loss = self.step(batch, "test")
        self.log("test/loss", loss)

        return loss

    @final
    def configure_optimizers(
        self: "BaseLitModule",
    ) -> tuple[list[Optimizer], list[dict[str, LRScheduler | str | int]]]:
        """.

        Returns:
            A tuple containing the PyTorch ``Optimizer`` and
            ``LRScheduler`` instance attributes (each nested in a
            list).
        """
        return [self.optimizer], [
            {
                "scheduler": self.lrscheduler,
                "interval": "step",
                "frequency": 1,
            },
        ]
