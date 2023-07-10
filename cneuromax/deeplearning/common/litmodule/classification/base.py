"""."""

from abc import ABCMeta
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
import torchmetrics
from beartype import beartype as typechecker
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.deeplearning.common.litmodule import BaseLitModule


@dataclass
class BaseClassificationLitModuleConfig:
    """.

    Attributes:
        num_classes: .
    """

    num_classes: int


class BaseClassificationLitModule(BaseLitModule, metaclass=ABCMeta):
    """.

    Attributes:
        accuracy (``torchmetrics.Accuracy``): The accuracy metric.
        config (``BaseClassificationLitModuleConfig``): .
    """

    def __init__(
        self: "BaseClassificationLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        lrscheduler: partial[LRScheduler],
        config: BaseClassificationLitModuleConfig,
    ) -> None:
        """Calls parent constructor and creates an accuracy metric.

        Args:
            nnmodule: .
            optimizer: .
            lrscheduler: .
            config: .
        """
        super().__init__(nnmodule, optimizer, lrscheduler)
        self.config = config
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.config.num_classes,
        )

    @typechecker
    def step(
        self: "BaseClassificationLitModule",
        batch: tuple[
            Float[Tensor, " batch_size *x_shape"],
            Int[Tensor, " batch_size"],
        ],
        stage: Literal["train", "val", "test"],
    ) -> Float[Tensor, " "]:
        """Computes accuracy and cross entropy loss.

        Args:
            batch: .
            stage: .

        Returns:
            The cross entropy loss.
        """
        # (BS x *XS, BS) -> BS x *XS, BS
        x, y = batch

        # BS x *XS -> BS x NC
        logits = self.nnmodule(x)

        # BS x NC -> BS
        preds = torch.argmax(logits, dim=1)

        # BS, BS -> BS
        accuracy = self.accuracy(preds, y)  # type: ignore[operator]
        # Torchmetrics Accuracy is not yet compatible with MyPy, see:
        # https://github.com/Lightning-AI/torchmetrics/blob/master/pyproject.toml

        # BS -> None
        self.log(f"{stage}/acc", accuracy)

        # BS x NC, BS -> []
        return F.cross_entropy(logits, y)
