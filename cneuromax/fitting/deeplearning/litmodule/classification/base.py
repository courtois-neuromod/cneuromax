""":class:`BaseClassificationLitModule` & its config."""

from abc import ABCMeta
from collections.abc import Callable  # noqa: TCH003
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Annotated as An
from typing import Any

import torch
import torch.nn.functional as f
import wandb
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.classification import MulticlassAccuracy

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.beartype import ge, one_of


@dataclass
class BaseClassificationLitModuleConfig:
    """Holds :class:`BaseClassificationLitModule` config values.

    Args:
        num_classes: Number of classes to classify between.
    """

    num_classes: An[int, ge(2)]


class BaseClassificationLitModule(BaseLitModule, metaclass=ABCMeta):
    """Base Classification :mod:`lightning` ``LitModule``.

    Args:
        config: See :class:`BaseClassificationLitModuleConfig`.
        nnmodule: See :paramref:`~.BaseLitModule.nnmodule`.
        optimizer: See :paramref:`~.BaseLitModule.optimizer`.
        scheduler: See :paramref:`~.BaseLitModule.scheduler`.

    Attributes:
        accuracy\
            (:class:`~torchmetrics.classification.MulticlassAccuracy`)
        wandb_input_data_wrapper (:`callable`): A wrapper to be used\
            around the input datapoint when logging to W&B.
        wandb_table: A W&B table to store validation data.
    """

    def __init__(
        self: "BaseClassificationLitModule",
        config: BaseClassificationLitModuleConfig,
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        super().__init__(
            nnmodule=nnmodule,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        # Accuracy metric.
        self.accuracy: MulticlassAccuracy = MulticlassAccuracy(
            num_classes=config.num_classes,
        )
        # W&B validation attributes.
        self.wandb_input_data_wrapper: Callable[..., Any] = lambda x: x
        self.wandb_table: wandb.Table = wandb.Table(  # type: ignore[no-untyped-call]
            columns=["idx", "epoch", "x", "y", "probs", "pred"],
        )

    def step(
        self: "BaseClassificationLitModule",
        batch: tuple[
            Float[Tensor, " batch_size *x_shape"],
            Int[Tensor, " batch_size"],
        ],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, " "]:
        """Computes the model accuracy and cross entropy loss.

        Args:
            batch: A tuple ``(X, y)`` where ``X`` is the input data and\
                ``y`` is the target data.
            stage: See\
                :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The cross entropy loss.
        """
        x: Float[Tensor, " batch_size *x_shape"] = batch[0]
        y: Int[Tensor, " batch_size"] = batch[1]
        logits: Float[Tensor, " batch_size num_classes"] = self.nnmodule(x)
        preds: Int[Tensor, " batch_size"] = torch.argmax(input=logits, dim=1)
        accuracy: Float[Tensor, " "] = self.accuracy(preds=preds, target=y)
        self.log(name=f"{stage}/acc", value=accuracy)
        if stage == "val":
            self.save_val_data(x=x, y=y, logits=logits, preds=preds)
        return f.cross_entropy(input=logits, target=y)

    def save_val_data(
        self: "BaseClassificationLitModule",
        x: Float[Tensor, " batch_size *x_shape"],
        y: Int[Tensor, " batch_size"],
        logits: Float[Tensor, " batch_size num_classes"],
        preds: Int[Tensor, " batch_size"],
    ) -> None:
        """Saves data computed during validation for later use.

        Args:
            x: The input data.
            y: The target data.
            logits: The network's raw output.
            preds: The model's predictions.
        """
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        probs = f.softmax(input=logits, dim=1).cpu().numpy()
        preds = preds.cpu().numpy()
        for x_i, y_i, probs_i, preds_i in zip(
            x,
            y,
            probs,
            preds,
            strict=False,
        ):
            self.val_data.append([x_i, y_i, probs_i, preds_i])

    def on_validation_epoch_end(self: "BaseClassificationLitModule") -> None:
        """Called at the end of the validation epoch."""
        for i, val_data in enumerate(self.val_data):
            x_i, y_i, probs_i, preds_i = val_data
            self.wandb_table.add_data(  # type: ignore[no-untyped-call]
                i,
                self.curr_val_epoch,
                self.wandb_input_data_wrapper(x_i),
                y_i,
                probs_i.tolist(),
                preds_i,
            )
        # 1) Static type checking discrepancy:
        # `logger.experiment` is a `wandb.wandb_run.Run` instance.
        # 2) Cannot log the same table twice:
        # https://github.com/wandb/wandb/issues/2981#issuecomment-1458447291
        self.logger.experiment.log({"val_data": copy(self.wandb_table)})  # type: ignore[union-attr]
        super().on_validation_epoch_end()
