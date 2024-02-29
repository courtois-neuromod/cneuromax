""":class:`WandbValLoggingMixin`."""

import logging
from collections.abc import Callable  # noqa: TCH003
from copy import copy
from typing import Any

import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from wandb.sdk.data_types.base_types.wb_value import WBValue


class WandbValLoggingMixin:
    """Mixin that logs val data to :mod:`wandb`.

    Args:
        activate: Whether to activate :mod:`wandb` validation\
            logging.
        logger: A :class:`~lightning.pytorch.loggers.wandb.WandbLogger`\
            instance.

    Attributes:
        logger (:class:`~lightning.pytorch.loggers.wandb.WandbLogger`):\
            See :paramref:`~WandbValLoggingMixin.logger`.
        is_on (`bool`): See :paramref:`~WandbValLoggingMixin.activate`.
        curr_val_epoch (`int`): The current validation epoch (can be\
            different from training epoch if validation is called\
            multiple times per training epoch).
        val_wandb_data (`list[list[Any]]`): A list of of lists\
            containing data relating to one specific example (ex:\
            `input_data`, `logits`, ...) meant to be logged to\
            :mod:`wandb`.
    """

    def __init__(
        self: "WandbValLoggingMixin",
        logger: Any,  # noqa: ANN401
        *,
        activate: bool,
    ) -> None:
        super().__init__()
        # Static type checking purposes only
        assert isinstance(logger, WandbLogger)  # noqa: S101
        self.logger = logger
        self.is_on = activate
        if self.is_on:
            wandb_columns = getattr(self, "wandb_columns", None)
            if not (isinstance(wandb_columns, list)):
                error_msg = (
                    "The `wandb_columns` attribute is either not defined or "
                    "not a list. Define it or turn off W&B validation logging."
                )
                raise TypeError(error_msg)
            self.curr_val_epoch = 0
            self.val_wandb_data: list[Any] = []
            self.wandb_table = wandb.Table(  # type: ignore[no-untyped-call]
                columns=[
                    "data_idx",
                    "val_epoch",
                    *wandb_columns,
                ],
            )
            if not (
                getattr(self, "wandb_x_wrapper")  # noqa: B009
                and isinstance(self.wandb_x_wrapper, WBValue)
            ):
                logging.warning(
                    "`wandb_x_wrapper` attribute not set/invalid. "
                    "Defaulting to not wrapping the input data.",
                )
                self.wandb_x_wrapper: Callable[..., Any] = lambda x: x

    def on_validation_start(self: "WandbValLoggingMixin") -> None:
        """Resets :attr:`val_wandb_data` if logging w/ :mod:`wandb`."""
        if self.is_on:
            self.val_wandb_data = []

    def on_validation_epoch_end(
        self: "WandbValLoggingMixin",
    ) -> None:
        """Uploads :attr:`val_wandb_data` if logging w/ :mod:`wandb`."""
        if self.is_on:
            for i, val_wandb_data_i in enumerate(self.val_wandb_data):
                self.wandb_table.add_data(  # type: ignore[no-untyped-call]
                    i,
                    self.curr_val_epoch,
                    *val_wandb_data_i,
                )
            # Cannot log the same table twice:
            # https://github.com/wandb/wandb/issues/2981#issuecomment-1458447291
            self.logger.experiment.log({"val_data": copy(self.wandb_table)})
            self.curr_val_epoch += 1
