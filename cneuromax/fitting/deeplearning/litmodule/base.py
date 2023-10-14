"""."""

from abc import ABCMeta
from functools import partial
from typing import Annotated as An
from typing import Any, final

from jaxtyping import Num
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.utils.annotations import one_of


class BaseLitModule(LightningModule, metaclass=ABCMeta):
    """Root Lightning ``Module`` class.

    Subclasses need to implement the ``step`` method that inputs a batch
    and returns the loss value(s).

    Attributes:
        nnmodule (PyTorch ``nn.Module``): .
        optimizer (PyTorch ``Optimizer``): .
        scheduler (PyTorch ``LRScheduler``): .
    """

    def __init__(
        self: "BaseLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        """Calls parent constructor & initializes instance attributes.

        Args:
            nnmodule: .
            optimizer: .
            scheduler: .
        """
        super().__init__()

        self.nnmodule: nn.Module = nnmodule
        self.optimizer: Optimizer = optimizer(params=self.parameters())
        self.scheduler: LRScheduler = scheduler(optimizer=self.optimizer)

    @final
    def x_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
        stage: An[str, one_of("train", "val", "test", "predict")],
    ) -> Num[Tensor, " ..."]:
        """Generic wrapper around the ``step`` instance method.

        Verifies that the ``step`` instance method is callable, calls
        it and logs the loss value(s).

        Args:
            batch: .
            stage: .

        Returns:
            The loss value(s).

        Raises:
            AttributeError: If the ``step`` instance method is not
                callable.
        """
        if not (hasattr(self, "step") and callable(self.step)):
            raise AttributeError

        if isinstance(batch, list):
            tupled_batch: tuple[Num[Tensor, " ..."], ...] = tuple(batch)

        loss: Num[Tensor, " ..."] = self.step(tupled_batch, stage)
        self.log(f"{stage}/loss", loss)

        return loss

    @final
    def training_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
    ) -> Num[Tensor, " ..."]:
        """Calls ``x_step`` method with argument ``train``.

        Returns:
            The loss value(s).
        """
        return self.x_step(batch, "train")

    @final
    def validation_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Num[Tensor, " ..."]:
        """Calls ``x_step`` method with argument ``val``.

        Args:
            batch: .
            *args: .
            **kwargs: .

        Returns:
            The loss value(s).
        """
        return self.x_step(batch, "val")

    @final
    def test_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
    ) -> Num[Tensor, " ..."]:
        """Calls ``x_step`` method with argument ``test``.

        Args:
            batch: .

        Returns:
            The loss value(s).
        """
        return self.x_step(batch, "test")

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
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        ]
