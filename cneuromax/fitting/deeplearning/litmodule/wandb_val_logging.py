""":class:`WandbValLoggingLightningModule`."""

import logging
from collections.abc import Callable  # noqa: TCH003
from copy import copy
from typing import Any

import wandb
from lightning.pytorch import LightningModule
from wandb.sdk.data_types.base_types.wb_value import WBValue


class WandbValLoggingLightningModule(LightningModule):
    """:class:`.LightningModule` that logs val data to :mod:`wandb`.

    TODO: Save/load attributes to/from checkpoint.

    Args:
        logs_val: Whether to activate :mod:`wandb` validation
            data logging.

    Attributes:
        logs_val (`bool`): See
            :paramref:`~WandbValLoggingMixin.logs_val`.
        curr_val_epoch (`int`): The current validation epoch (can be
            different from training epoch if validation is called
            multiple times per training epoch).
        val_wandb_data (`list[list[Any]]`): A list of dictionaries
            containing validation data relating to one specific example
            (ex: `input_data`, `logits`, ...) meant to be logged to
            :mod:`wandb`.
        wandb_columns (`list[str]`): A list of strings representing
            the keys of the dictionaries in :attr:`val_wandb_data`.
        wandb_table (:class:`~wandb.Table`): A table to upload to W&B
            containing validation data.
        wandb_x_wrapper (`Callable`): A callable that wraps the input
            data before logging it to W&B.
    """

    def __init__(
        self: "WandbValLoggingLightningModule",
        *,
        logs_val: bool,
    ) -> None:
        super().__init__()
        self.logs_val = logs_val

    def on_fit_start(self: "WandbValLoggingLightningModule") -> None:
        """Instantiates :mod:`wandb` attributes if :attr:`logs_val`."""
        if self.logs_val:
            self.curr_val_epoch = 0
            self.val_wandb_data: list[dict[str, Any]] = []
            wandb_columns = getattr(self, "wandb_columns", None)
            if not (isinstance(wandb_columns, list)):
                error_msg = (
                    "The `wandb_columns` attribute is either not defined or "
                    "not a list. Define it or turn off W&B validation logging."
                )
                raise TypeError(error_msg)
            self.wandb_table = wandb.Table(  # type: ignore[no-untyped-call]
                columns=["data_idx", "val_epoch", *wandb_columns],
            )
            if not (
                getattr(self, "wandb_x_wrapper")  # noqa: B009
                and isinstance(self.wandb_x_wrapper, type(WBValue))
            ):
                logging.warning(
                    "`wandb_x_wrapper` attribute not set/invalid. "
                    "Defaulting to not wrapping the input data.",
                )
                self.wandb_x_wrapper: Callable[..., Any] = lambda x: x

    def on_validation_start(self: "WandbValLoggingLightningModule") -> None:
        """Resets :attr:`val_wandb_data` if :attr:`logs_val`."""
        if self.logs_val:
            self.val_wandb_data = []

    def on_validation_epoch_end(
        self: "WandbValLoggingLightningModule",
    ) -> None:
        """Uploads :attr:`val_wandb_data` if :attr:`logs_val`."""
        if self.logs_val:
            for i, val_wandb_data_i in enumerate(self.val_wandb_data):
                self.wandb_table.add_data(  # type: ignore[no-untyped-call]
                    i,
                    self.curr_val_epoch,
                    *[val_wandb_data_i[key] for key in self.wandb_columns],
                )
            # 1) Static type checking discrepancy:
            # `logger.experiment` is a `wandb.wandb_run.Run` instance.
            # 2) Cannot log the same table twice:
            # https://github.com/wandb/wandb/issues/2981#issuecomment-1458447291
            try:
                self.logger.experiment.log(  # type: ignore[union-attr]
                    {"val_data": copy(self.wandb_table)},
                )
            except Exception as e:
                error_msg = (
                    "Failed to log validation data to W&B. "
                    "You might be trying to log tensors."
                )
                raise ValueError(error_msg) from e
            self.curr_val_epoch += 1
