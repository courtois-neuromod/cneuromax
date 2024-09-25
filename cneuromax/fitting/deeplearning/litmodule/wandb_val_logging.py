""":class:`WandbValLoggingLightningModule`."""

from copy import copy
from typing import Any

import wandb
from lightning.pytorch import LightningModule


class WandbValLoggingLightningModule(LightningModule):
    """:class:`.LightningModule` that logs val data to ``wandb``.

    Ref: `wandb <https://wandb.ai/>`_

    TODO: Save/load attributes to/from checkpoint.

    Args:
        logs_val: Whether to activate `wandb <https://wandb.ai/>`_
            validation data logging.

    Attributes:
        logs_val (`bool`): See
            :paramref:`~WandbValLoggingMixin.logs_val`.
        curr_val_epoch (`int`): The current validation epoch (can be
            different from training epoch if validation is called
            multiple times per training epoch).
        wandb_table (:class:`~wandb.Table`): A table to upload to W&B
            containing validation data.
        wandb_columns (`list[str]`): A list of strings representing
            the keys of the dictionaries in :attr:`val_wandb_data`.
        val_wandb_data (`list[list[Any]]`): A list of dictionaries
            containing validation data relating to one specific example
            (ex: `input_data`, `logits`, ...) meant to be logged in
            :attr:`val_wandb_data`.
    """

    def __init__(
        self: "WandbValLoggingLightningModule",
        *,
        logs_val: bool,
    ) -> None:
        super().__init__()
        self.logs_val = logs_val

    def on_fit_start(self: "WandbValLoggingLightningModule") -> None:
        """Instantiates ``wandb`` attributes if :attr:`logs_val`.

        Ref: `wandb <https://wandb.ai/>`_
        """
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
