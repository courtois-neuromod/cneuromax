""":class:`BaseLitModule` & its config."""

import logging
from abc import ABCMeta
from collections.abc import Callable  # noqa: TCH003
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Annotated as An
from typing import Any, final

import wandb
from jaxtyping import Num
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from wandb.sdk.data_types.base_types.wb_value import WBValue

from cneuromax.fitting.deeplearning.utils.type import Batch_type
from cneuromax.utils.beartype import one_of


@dataclass
class BaseLitModuleConfig:
    """Holds :class:`BaseLitModule` config values.

    Args:
        log_val_wandb: Whether to log validation data to :mod:`wandb`.
    """

    log_val_wandb: bool = False


class BaseLitModule(LightningModule, metaclass=ABCMeta):
    """Base :mod:`lightning` ``LitModule``.

    Subclasses need to implement the :meth:`step` method that inputs
    both ``batch`` (``tuple[torch.Tensor]``) and  ``stage`` (``str``)
    arguments while returning the loss value(s) in the form of a
    :class:`torch.Tensor`.

    Example definition:

    .. highlight:: python
    .. code-block:: python

        def step(
            self: "BaseClassificationLitModule",
            batch: tuple[
                Float[Tensor, " batch_size *x_shape"],
                Int[Tensor, " batch_size"],
            ],
            stage: An[str, one_of("train", "val", "test")],
        ) -> Float[Tensor, " "]:
            ...

    Note:
        ``batch`` and loss value(s) type hints in this class are not
        rendered properly in the documentation due to an\
        incompatibility between :mod:`sphinx` and :mod:`jaxtyping`.\
        Refer to the source code available next to the method\
        signatures to find the correct types.

    Args:
        config: See :class:`BaseLitModuleConfig`.
        nnmodule: A :mod:`torch` ``nn.Module`` to be used by this\
            instance.
        optimizer: A :mod:`torch` ``Optimizer`` to be used by this\
            instance. It is partial as an argument as the\
            :paramref:`nnmodule` parameters are required for its\
            initialization.
        scheduler: A :mod:`torch` ``Scheduler`` to be used by this\
            instance. It is partial as an argument as the\
            :paramref:`optimizer` is required for its initialization.

    Attributes:
        config (:class:`BaseLitModuleConfig`): See\
            :paramref:`~BaseLitModule.config`.
        nnmodule (:class:`torch.nn.Module`): See\
            :paramref:`~BaseLitModule.nnmodule`.
        optimizer (:class:`torch.optim.Optimizer`): See\
            :paramref:`~BaseLitModule.optimizer`.
        scheduler (:class:`torch.optim.lr_scheduler.LRScheduler`): See\
            :paramref:`~BaseLitModule.scheduler`.
        val_wandb_data (`list`): A list of data collected during\
            validation for logging to :mod:`wandb`.
        curr_val_epoch (`int`): The current validation epoch.


    Raises:
        NotImplementedError: If the :meth:`step` method is not\
            defined or not callable.
    """

    def __init__(
        self: "BaseLitModule",
        config: BaseLitModuleConfig,
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        super().__init__()
        self.config = config
        self.instantiate_torch_attributes(
            nnmodule=nnmodule,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        # Initialize W&B validation logging attributes
        if config.log_val_wandb:
            self.instantiate_wandb_attributes()
        # Verify `step` method.
        if not callable(getattr(self, "step", None)):
            error_msg = (
                "The `BaseLitModule.step` method is not defined/not callable."
            )
            raise NotImplementedError(error_msg)

    def instantiate_torch_attributes(
        self: "BaseLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        """Instantiates :mod:`torch` attributes.

        Args:
            nnmodule: See :paramref:`~BaseLitModule.nnmodule`.
            optimizer: See :paramref:`~BaseLitModule.optimizer`.
            scheduler: See :paramref:`~BaseLitModule.scheduler`.
        """
        self.nnmodule = nnmodule
        self.optimizer = optimizer(params=self.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)

    def instantiate_wandb_attributes(self: "BaseLitModule") -> None:
        """Instantiates W&B attributes."""
        self.curr_val_epoch = 0
        self.val_wandb_data: list[Any] = []
        if not (isinstance(getattr(self, "wandb_columns", None), list)):
            error_msg = (
                "The `wandb_columns` attribute is either not defined or "
                "not a list. Define it or turn off W&B validation logging."
            )
            raise TypeError(error_msg)
        self.wandb_table = wandb.Table(  # type: ignore[no-untyped-call]
            columns=[
                "data_idx",
                "val_epoch",
                *self.wandb_columns,
            ],
        )

    @final
    def stage_step(
        self: "BaseLitModule",
        batch: Batch_type,
        stage: An[str, one_of("train", "val", "test", "predict")],
    ) -> Num[Tensor, " ..."]:
        """Generic stage wrapper around the :meth:`step` method.

        Verifies that the :meth:`step` method exists and is callable,
        calls it and logs the loss value(s).

        Args:
            batch: The batched input data.
            stage: The current stage (``train``, ``val``, ``test`` or\
                ``predict``).

        Returns:
            The loss value(s).
        """
        if isinstance(batch, list):
            batch = tuple(batch)
        loss: Num[Tensor, " ..."] = self.step(batch, stage)
        self.log(name=f"{stage}/loss", value=loss)
        return loss

    @final
    def training_step(
        self: "BaseLitModule",
        batch: Batch_type,
    ) -> Num[Tensor, " ..."]:
        """Calls :meth:`stage_step` with argument ``stage="train"``.

        Args:
            batch: See :paramref:`~stage_step.batch`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(batch=batch, stage="train")

    @final
    def validation_step(
        self: "BaseLitModule",
        batch: Batch_type,
        # :paramref:`*args` & :paramref:`**kwargs` type annotations
        # cannot be more specific because of
        # :meth:`LightningModule.validation_step`\'s signature.
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Num[Tensor, " ..."]:
        """Calls :meth:`stage_step` with argument ``stage="val"``.

        Args:
            batch: See :paramref:`~stage_step.batch`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The loss value(s).
        """
        return self.stage_step(batch=batch, stage="val")

    def on_validation_start(self: "BaseLitModule") -> None:
        """Resets :attr:`val_wandb_data` if logging w/ :mod:`wandb`."""
        if self.config.log_val_wandb:
            self.val_wandb_data = []

    def on_validation_epoch_end(self: "BaseLitModule") -> None:
        """Uploads :attr:`val_wandb_data` if logging w/ :mod:`wandb`."""
        if self.config.log_val_wandb:
            for i, val_wandb_data_i in enumerate(self.val_wandb_data):
                self.wandb_table.add_data(  # type: ignore[no-untyped-call]
                    i,
                    self.curr_val_epoch,
                    *val_wandb_data_i,
                )
            # 1) Static type checking discrepancy:
            # `logger.experiment` is a `wandb.wandb_run.Run` instance.
            # 2) Cannot log the same table twice:
            # https://github.com/wandb/wandb/issues/2981#issuecomment-1458447291
            self.logger.experiment.log({"val_data": copy(self.wandb_table)})  # type: ignore[union-attr]
            self.curr_val_epoch += 1

    @final
    def test_step(
        self: "BaseLitModule",
        batch: Batch_type,
    ) -> Num[Tensor, " ..."]:
        """Calls :meth:`stage_step` with argument ``stage="test"``.

        Args:
            batch: See :paramref:`~stage_step.batch`.

        Returns:
            The loss value(s).
        """
        return self.stage_step(batch=batch, stage="test")

    @final
    def configure_optimizers(
        self: "BaseLitModule",
    ) -> tuple[list[Optimizer], list[dict[str, LRScheduler | str | int]]]:
        """Returns a dict with :attr:`optimizer` and :attr:`scheduler`.

        Returns:
            A tuple containing this instance's\
            :class:`~torch.optim.Optimizer` and\
            :class:`~torch.optim.lr_scheduler.LRScheduler`\
            attributes.
        """
        return [self.optimizer], [
            {"scheduler": self.scheduler, "interval": "step", "frequency": 1},
        ]
