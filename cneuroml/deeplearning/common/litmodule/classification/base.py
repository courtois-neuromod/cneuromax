"""."""
from abc import ABCMeta
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

from cneuroml.deeplearning.common.litmodule import BaseLitModule


class BaseClasssificationLitModule(BaseLitModule, metaclass=ABCMeta):
    """.

    Attributes:
        accuracy (``torchmetrics.Accuracy``): The accuracy metric.
        nnmodule (``nn.Module``): .
        optimizer (``Optimizer``): .
        scheduler (``LRScheduler``): .
    """

    def __init__(
        self: "BaseClasssificationLitModule",
        nnmodule: nn.Module,
        optimizer_partial: partial[Optimizer],
        scheduler_partial: partial[LRScheduler],
        num_classes: int,
    ) -> None:
        """Calls parent constructor and creates an accuracy metric.

        Args:
            nnmodule: .
            optimizer_partial: .
            scheduler_partial: .
            num_classes: .
        """
        super().__init__(nnmodule, optimizer_partial, scheduler_partial)

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

    @typechecker
    def step(
        self: "BaseClasssificationLitModule",
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
