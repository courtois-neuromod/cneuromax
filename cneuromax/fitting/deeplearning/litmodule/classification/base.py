""":class:`BaseClassificationLitModule` & its config."""

import logging
from abc import ABCMeta
from collections.abc import Callable  # noqa: TCH003
from dataclasses import dataclass
from typing import Annotated as An
from typing import Any

import torch
import torch.nn.functional as f
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy
from wandb.sdk.data_types.base_types.wb_value import WBValue

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
    BaseLitModuleConfig,
)
from cneuromax.utils.beartype import ge, one_of


@dataclass
class BaseClassificationLitModuleConfig(BaseLitModuleConfig):
    """Holds :class:`BaseClassificationLitModule` config values.

    Args:
        num_classes: Number of classes to classify between.
    """

    num_classes: An[int, ge(2)] = 2


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
        wandb_table (:class:`~wandb.Table`): A table to upload to W&B\
            containing validation data.

        wandb_input_data_wrapper (:`callable`): A wrapper to be used\
            around the input datapoint when logging to W&B.
    """

    def __init__(
        self: "BaseClassificationLitModule",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config: BaseClassificationLitModuleConfig
        # Accuracy metric.
        self.accuracy: MulticlassAccuracy = MulticlassAccuracy(
            num_classes=self.config.num_classes,
        )
        # W&B validation attributes.
        if self.config.log_val_wandb:
            self.wandb_columns = ["x", "y", "y_hat_probs", "y_hat"]
            if not (
                getattr(self, "wandb_x_wrapper")  # noqa: B009
                and isinstance(self.wandb_x_wrapper, WBValue)
            ):
                logging.warning(
                    "`wandb_x_wrapper` attribute not set/invalid. "
                    "Defaulting to no wrapper.",
                )
                self.wandb_x_wrapper: Callable[..., Any] = lambda x: x

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
        y_hat: Int[Tensor, " batch_size"] = torch.argmax(input=logits, dim=1)
        accuracy: Float[Tensor, " "] = self.accuracy(preds=y_hat, target=y)
        self.log(name=f"{stage}/acc", value=accuracy)
        if stage == "val" and self.config.log_val_wandb:
            self.save_val_data(x=x, y=y, logits=logits, y_hat=y_hat)
        return f.cross_entropy(input=logits, target=y)

    def save_val_data(
        self: "BaseClassificationLitModule",
        x: Float[Tensor, " batch_size *x_shape"],
        y: Int[Tensor, " batch_size"],
        logits: Float[Tensor, " batch_size num_classes"],
        y_hat: Int[Tensor, " batch_size"],
    ) -> None:
        """Saves data computed during validation for later use.

        Args:
            x: The input data.
            y: The target class.
            logits: The network's raw `num_classes` output.
            y_hat: The model's predicted class.
        """
        for x_i, y_i, y_hat_i, y_hat_probs_i in zip(
            x.cpu(),
            y.cpu(),
            y_hat.cpu(),
            f.softmax(input=logits, dim=1).cpu(),
            strict=False,
        ):
            self.val_wandb_data.append(
                [
                    self.wandb_x_wrapper(x_i),
                    y_i,
                    y_hat_i,
                    y_hat_probs_i.tolist(),
                ],
            )
