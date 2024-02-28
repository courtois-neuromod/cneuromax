""":class:`BaseAutoencodingLitModule`."""

from collections.abc import Callable  # noqa: TCH003
from copy import copy
from dataclasses import dataclass
from typing import Annotated as An
from typing import Any

import torch.nn.functional as f
import wandb
from jaxtyping import Float
from torch import Tensor

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.fitting.deeplearning.litmodule.autoencoding.nnmodule import (
    BaseAutoEncoder,
)
from cneuromax.utils.beartype import one_of


@dataclass
class BaseAutoencodingLitModuleConfig:
    """Holds :class:`BaseAutoencodingLitModule` config values.

    Args:
        log_val_preds: Whether to log validation predictions to W&B.
    """

    log_val_preds: bool = False


class BaseAutoencodingLitModule(BaseLitModule):
    """Base Autoencoding :class:`.BaseLitModule`.

    Args:
        nnmodule: See :class:`.BaseAutoEncoder`.
    """

    def __init__(
        self: "BaseAutoencodingLitModule",
        config: BaseAutoencodingLitModuleConfig,
        nnmodule: BaseAutoEncoder,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(nnmodule, *args, **kwargs)
        self.config = config
        # W&B validation attributes.
        self.wandb_input_data_wrapper: Callable[..., Any] = lambda x: x
        self.wandb_table = wandb.Table(  # type: ignore[no-untyped-call]
            columns=["idx", "epoch", "x", "pred"],
        )

    def step(
        self: "BaseAutoencodingLitModule",
        batch: Float[Tensor, " batch_size *data_dim"],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, ""]:
        """Computes the mean squared error loss for the autoencoder.

        Args:
            batch: The input data batch.
            stage: See\
                :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The mean squared error loss.
        """
        x = batch
        recon_x = self.nnmodule(x)
        recon_loss = f.mse_loss(x, recon_x)
        self.log(name=f"{stage}/mse", value=recon_loss)
        if stage == "val" and self.config.log_val_preds:
            self.save_val_data(x=x, preds=recon_x)
        return recon_loss

    def save_val_data(
        self: "BaseAutoencodingLitModule",
        x: Float[Tensor, " batch_size *data_dim"],
        preds: Float[Tensor, " batch_size *data_dim"],
    ) -> None:
        """Saves data computed during validation for later use.

        Args:
            x: The input data.
            y: The target data.
            logits: The network's raw output.
            preds: The model's predictions.
        """
        x = x.cpu().numpy()
        preds = preds.cpu().numpy()
        for x_i, preds_i in zip(
            x,
            preds,
            strict=False,
        ):
            self.val_data.append([x_i, preds_i])

    def on_validation_epoch_end(self: "BaseAutoencodingLitModule") -> None:
        """Called at the end of the validation epoch."""
        if not self.config.log_val_preds:
            return
        for i, val_data in enumerate(self.val_data):
            x_i, preds_i = val_data
            self.wandb_table.add_data(  # type: ignore[no-untyped-call]
                i,
                self.curr_val_epoch,
                self.wandb_input_data_wrapper(x_i),
                self.wandb_input_data_wrapper(preds_i),
            )
        # 1) Static type checking discrepancy:
        # `logger.experiment` is a `wandb.wandb_run.Run` instance.
        # 2) Cannot log the same table twice:
        # https://github.com/wandb/wandb/issues/2981#issuecomment-1458447291
        self.logger.experiment.log({"val_data": copy(self.wandb_table)})  # type: ignore[union-attr]
        super().on_validation_epoch_end()
