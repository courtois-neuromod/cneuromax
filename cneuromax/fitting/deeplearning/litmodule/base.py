"""Base :mod:`cneuromax.fitting.deeplearning` LitModule Class."""

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
    """Root :mod:`~lightning.pytorch.LightningModule` subclass.

    Subclasses need to implement the :meth:`step` method that inputs a
    (tupled) batch and returns the loss value(s).

    Attributes:
        nnmodule (:class:`torch.nn.Module`): This instance's\
            :class:`~torch.nn.Module`.
        optimizer (:class:`torch.optim.Optimizer`): This instance's\
            :class:`~torch.optim.Optimizer`.
        scheduler (:class:`torch.optim.lr_scheduler.LRScheduler`): This\
            instance's :class:`~torch.optim.lr_scheduler.LRScheduler`.
    """

    def __init__(
        self: "BaseLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        """Calls parent constructor & initializes instance attributes.

        Args:
            nnmodule: A :class:`~torch.nn.Module` instance.
            optimizer: A :class:`~torch.optim.Optimizer` partial.
            scheduler: A\
                :class:`~torch.optim.lr_scheduler.LRScheduler` partial.
        """
        super().__init__()
        self.nnmodule = nnmodule
        self.optimizer = optimizer(params=self.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)

        if not callable(getattr(self, "step", None)):
            raise NotImplementedError

    @final
    def stage_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
        stage: An[str, one_of("train", "val", "test", "predict")],
    ) -> Num[Tensor, " ..."]:
        """Generic stage wrapper around the `step` instance method.

        Verifies that the `step` instance method exists and is callable,
        calls it and logs the loss value(s).

        Args:
            batch: The input data batch.
            stage: The current stage.

        Returns:
            loss: The loss value(s).

        Raises:
            :class:`AttributeError`: If the `step` instance method is\
                not callable.
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
        """Calls :meth:`stage_step` method with argument `stage=train`.

        Args:
            batch: See\
                :paramref:`~BaseLitModule.stage_step.batch`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(batch=batch, stage="train")

    @final
    def validation_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Num[Tensor, " ..."]:
        """Calls :meth:`stage_step` method with argument `stage=val`.

        `*args` and `**kwargs` type annotations cannot be more
        specific, because of
        :meth:`~lightning.pytorch.LightningModule.validation_step`
        method's signature.

        Args:
            batch: See\
                :paramref:`~BaseLitModule.stage_step.batch`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The loss value(s).
        """
        return self.stage_step(batch=batch, stage="val")

    @final
    def test_step(
        self: "BaseLitModule",
        batch: Num[Tensor, " ..."]
        | tuple[Num[Tensor, " ..."], ...]
        | list[Num[Tensor, " ..."]],
    ) -> Num[Tensor, " ..."]:
        """Calls :meth:`stage_step` method with argument `stage=test`.

        Args:
            batch: See\
                :paramref:`~BaseLitModule.stage_step.batch`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(batch=batch, stage="test")

    @final
    def configure_optimizers(
        self: "BaseLitModule",
    ) -> tuple[list[Optimizer], list[dict[str, LRScheduler | str | int]]]:
        """Returns a dictionary with the optimizer and scheduler.

        Returns:
            A tuple containing the\
            :class:`~torch.optim.Optimizer` and\
            :class:`~torch.optim.lr_scheduler.LRScheduler` instance\
            attributes (each nested in a list).
        """
        return [self.optimizer], [
            {"scheduler": self.scheduler, "interval": "step", "frequency": 1},
        ]
