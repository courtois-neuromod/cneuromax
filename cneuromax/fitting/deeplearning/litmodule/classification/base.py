"""."""

from abc import ABCMeta
from functools import partial
from typing import Annotated as An

import torch
import torch.nn.functional as f
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.classification import MulticlassAccuracy

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.annotations import ge, one_of


class BaseClassificationLitModuleConfig:
    """Base classification Lightning Module config.

    Attributes:
        num_classes: .
    """

    num_classes: An[int, ge(2)]


class BaseClassificationLitModule(BaseLitModule, metaclass=ABCMeta):
    """Base classification Lightning Module.

    Attributes:
        accuracy (``MulticlassAccuracy``): The accuracy metric.
        config (``BaseClassificationLitModuleConfig``): .
    """

    def __init__(
        self: "BaseClassificationLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
        num_classes: An[int, ge(2)],
    ) -> None:
        """Calls parent constructor & initializes accuracy metric.

        Args:
            nnmodule: .
            optimizer: .
            scheduler: .
            num_classes: .
        """
        super().__init__(nnmodule, optimizer, scheduler)
        self.accuracy: MulticlassAccuracy = MulticlassAccuracy(
            num_classes=num_classes,
        )

    def step(
        self: "BaseClassificationLitModule",
        batch: tuple[
            Float[Tensor, " batch_size *x_shape"],
            Int[Tensor, " batch_size"],
        ],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, " "]:
        """Computes accuracy and cross entropy loss.

        Args:
            batch: .
            stage: .

        Returns:
            The cross entropy loss.
        """
        x: Float[Tensor, " batch_size *x_shape"] = batch[0]
        y: Int[Tensor, " batch_size"] = batch[1]
        logits: Float[Tensor, " batch_size num_classes"] = self.nnmodule(x)
        preds: Int[Tensor, " batch_size"] = torch.argmax(logits, dim=1)
        accuracy: Float[Tensor, " "] = self.accuracy(preds, y)
        self.log(f"{stage}/acc", accuracy)
        return f.cross_entropy(logits, y)
