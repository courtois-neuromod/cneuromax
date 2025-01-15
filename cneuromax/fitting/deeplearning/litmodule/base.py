""":class:`.BaseLitModule` & its config."""

from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Annotated as An
from typing import Any, final

from jaxtyping import Num
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRSchedulerConfig,
)
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.utils.beartype import one_of

from .wandb_val_logging import WandbValLoggingLightningModule


@dataclass
class BaseLitModuleConfig:
    """Holds :class:`BaseLitModule` config values.

    Args:
        log_val_wandb: Whether to log validation data to
            `W&B <https://wandb.ai/>`_.
    """

    log_val_wandb: bool = False


class BaseLitModule(WandbValLoggingLightningModule, ABC):
    """Base :class:`lightning.pytorch.core.LightningModule`.

    Subclasses need to implement the :meth:`step` method that inputs
    both ``data`` and  ``stage`` arguments while returning the loss
    value(s) in the form of a :class:`torch.Tensor`.

    Example definition:

    .. highlight:: python
    .. code-block:: python

        def step(
            self: "BaseClassificationLitModule",
            data: tuple[
                Float[Tensor, " batch_size *x_dim"],
                Int[Tensor, " batch_size"],
            ],
            stage: An[str, one_of("train", "val", "test")],
        ) -> Float[Tensor, " "]:
            ...

    Note:
        ``data`` and loss value(s) type hints in this class are not \
        rendered properly in the documentation due to an \
        incompatibility between `sphinx <https://www.sphinx-doc.org/>`_
        and `jaxtyping <https://docs.kidger.site/jaxtyping/>`_. \
        Refer to the source code available next to the method \
        signatures to find the correct types.

    Args:
        config
        nnmodule: A :mod:`torch` ``nn.Module`` to be used by this
            instance.
        optimizer: A :mod:`torch` ``Optimizer`` to be used by this
            instance. It is partial as an argument as the
            :paramref:`nnmodule` parameters are required for its
            initialization.
        scheduler: A :mod:`torch` ``Scheduler`` to be used by this
            instance. It is partial as an argument as the
            :paramref:`optimizer` is required for its initialization.

    Attributes:
        config (BaseLitModuleConfig): See
            :paramref:`~BaseLitModule.config`.
        nnmodule (torch.nn.Module): See
            :paramref:`~BaseLitModule.nnmodule`.
        optimizer_partial (~functools.partial[torch.optim.Optimizer]):
            See :paramref:`~BaseLitModule.optimizer`.
        scheduler_partial (~functools.partial[\
            torch.optim.lr_scheduler.LRScheduler]): See
            :paramref:`~BaseLitModule.scheduler`.
        optimizer (torch.optim.Optimizer):
            :paramref:`~BaseLitModule.optimizer` instantiated.
        scheduler (torch.optim.lr_scheduler.LRScheduler):
            :paramref:`~BaseLitModule.scheduler` instantiated.

    Raises:
        NotImplementedError: If the :meth:`step` method is not
            defined or callable.
    """

    def __init__(
        self: "BaseLitModule",
        config: BaseLitModuleConfig,
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        super().__init__(logs_val=config.log_val_wandb)
        self.config = config
        self.nnmodule = nnmodule
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        # Verify `step` method.
        if not callable(getattr(self, "step", None)):
            error_msg = (
                "The `BaseLitModule.step` method is not defined/not callable."
            )
            raise NotImplementedError(error_msg)

    @final
    def stage_step(
        self: "BaseLitModule",
        data: Any,  # noqa: ANN401
        stage: An[str, one_of("train", "val", "test", "predict")],
    ) -> Num[Tensor, " *_"]:
        """Generic stage wrapper around the :meth:`step` method.

        Verifies that the :meth:`step` method exists and is callable,
        calls it and logs the loss value(s).

        Args:
            data: The batched input data.
            stage: The current stage (``train``, ``val``, ``test`` or
                ``predict``).

        Returns:
            The loss value(s).
        """
        if isinstance(data, list):
            data = tuple(data)
        loss: Num[Tensor, " *_"] = self.step(data, stage)
        self.log(name=f"{stage}_step/loss", value=loss, on_step=True)
        self.log(name=f"{stage}_epoch/loss", value=loss, on_epoch=True)
        return loss

    @final
    def training_step(
        self: "BaseLitModule",
        data: Any,  # noqa: ANN401
    ) -> Num[Tensor, " *_"]:
        """Calls :meth:`stage_step` with argument ``stage="train"``.

        Args:
            data: See :paramref:`~stage_step.data`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(data=data, stage="train")

    @final
    def validation_step(
        self: "BaseLitModule",
        data: Any,  # noqa: ANN401
        # :paramref:`*args` & :paramref:`**kwargs` type annotations
        # cannot be more specific because of
        # :meth:`LightningModule.validation_step`\'s signature.
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Num[Tensor, " *_"]:
        """Calls :meth:`stage_step` with argument ``stage="val"``.

        Args:
            data: See :paramref:`~stage_step.data`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(data=data, stage="val")

    @final
    def test_step(
        self: "BaseLitModule",
        data: Any,  # noqa: ANN401
    ) -> Num[Tensor, " *_"]:
        """Calls :meth:`stage_step` with argument ``stage="test"``.

        Args:
            data: See :paramref:`~stage_step.data`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(data=data, stage="test")

    def configure_optimizers(
        self: "BaseLitModule",
    ) -> OptimizerLRSchedulerConfig:
        """Returns a dict with :attr:`optimizer` and :attr:`scheduler`.

        Returns:
            This instance's
                :attr:`~torch.optim.Optimizer` and
                :class:`torch.optim.lr_scheduler.LRScheduler`.
        """
        self.optimizer = self.optimizer_partial(params=self.parameters())
        self.scheduler = self.scheduler_partial(optimizer=self.optimizer)
        return OptimizerLRSchedulerConfig(
            optimizer=self.optimizer,
            lr_scheduler=LRSchedulerConfigType(
                scheduler=self.scheduler,
                interval="step",
                frequency=1,
                reduce_on_plateau=False,
            ),
        )
