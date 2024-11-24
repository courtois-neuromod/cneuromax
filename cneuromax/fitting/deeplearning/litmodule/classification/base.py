""":class:`.BaseClassificationLitModule` & its config."""

from abc import ABC
from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass
from typing import Annotated as An
from typing import Any

import torch
import torch.nn.functional as f
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy

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


class BaseClassificationLitModule(BaseLitModule, ABC):
    """Base Classification ``LightningModule``.

    Ref: :class:`lightning.pytorch.core.LightningModule`

    If logging validation data to W&B, make sure to define the
    :attr:`wandb_columns` attribute in the subclass.

    Attributes:
        config (BaseClassificationLitModuleConfig)
        accuracy (torchmetrics.classification.MulticlassAccuracy)
        wandb_table (wandb.Table): A table to upload to W&B
            containing validation data.
    """

    def __init__(
        self: "BaseClassificationLitModule",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config: BaseClassificationLitModuleConfig
        self.accuracy: MulticlassAccuracy = MulticlassAccuracy(
            num_classes=self.config.num_classes,
        )
        if self.config.log_val_wandb:
            self.wandb_columns = ["x", "y", "y_hat", "logits"]
            self.wandb_input_data_wrapper: Callable[..., Any] = lambda x: x

    def step(
        self: "BaseClassificationLitModule",
        data: tuple[
            Float[Tensor, " batch_size *x_dim"],
            Int[Tensor, " batch_size"],
        ],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, " "]:
        """Computes the model accuracy and cross entropy loss.

        Args:
            data: A tuple ``(x, y)`` where ``x`` is the input data and
                ``y`` is the target data.
            stage: See
                :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The cross entropy loss.
        """
        x: Float[Tensor, " batch_size *x_dim"] = data[0]
        y: Int[Tensor, " batch_size"] = data[1]
        logits: Float[Tensor, " batch_size num_classes"] = self.nnmodule(x)
        y_hat: Int[Tensor, " batch_size"] = torch.argmax(input=logits, dim=1)
        accuracy: Float[Tensor, " "] = self.accuracy(preds=y_hat, target=y)
        self.log(name=f"{stage}/acc", value=accuracy)
        if stage == "val" and self.config.log_val_wandb:
            self.save_val_data(x=x, y=y, y_hat=y_hat, logits=logits)
        return f.cross_entropy(input=logits, target=y)

    def save_val_data(
        self: "BaseClassificationLitModule",
        x: Float[Tensor, " batch_size *x_dim"],
        y: Int[Tensor, " batch_size"],
        y_hat: Int[Tensor, " batch_size"],
        logits: Float[Tensor, " batch_size num_classes"],
    ) -> None:
        """Saves data computed during validation for later use.

        Args:
            x: The input data.
            y: The target class.
            y_hat: The predicted class.
            logits: The raw `num_classes` network outputs.
        """
        for x_i, y_i, y_hat_i, logits_i in zip(
            x.cpu(),
            y.cpu(),
            y_hat.cpu(),
            logits.cpu(),
            strict=False,
        ):
            self.val_wandb_data.append(
                {
                    "x": self.wandb_input_data_wrapper(x_i),
                    "y": y_i,
                    "y_hat": y_hat_i,
                    "logits": logits_i.tolist(),
                },
            )
