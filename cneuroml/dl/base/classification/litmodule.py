"""Base Classification LitModule.

Abbreviations used in this module:

PyTorch ``nn.Module`` is short for ``torch.nn.Module``.

PyTorch ``Optimizer`` is short for ``torch.optim.Optimizer``.

PyTorch ``LRScheduler`` is short for
``torch.optim.lr_scheduler.LRScheduler``.

``Float`` is short for ``jaxtyping.Float``.
"""

from abc import ABCMeta
from functools import partial

import torch
import torch.nn.functional as F
import torchmetrics
from beartype import beartype as typechecker
from jaxtyping import Float
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuroml.dl.base.litmodule import BaseLitModule


class BaseClassificationLitModule(BaseLitModule, metaclass=ABCMeta):
    """The Base Classification LitModule class.

    Attributes:
        accuracy (``torchmetrics.Accuracy``): The accuracy metric.
        nnmodule (``nn.Module``): The PyTorch ``nn.Module`` instance.
        optimizer (``Optimizer``): The PyTorch ``Optimizer`` instance.
        scheduler (``LRScheduler``): The PyTorch ``LRScheduler``
            instance.
    """

    def __init__(
        self: "BaseClassificationLitModule",
        nnmodule: nn.Module,
        optimizer_partial: partial[Optimizer],
        scheduler_partial: partial[LRScheduler],
        num_classes: int,
    ) -> None:
        """Constructor.

        Calls parent constructor and creates the accuracy metric.

        Args:
            nnmodule: A PyTorch ``nn.Module`` instance.
            optimizer_partial: A PyTorch ``Optimizer`` partial function.
            scheduler_partial: A PyTorch ``LRScheduler`` partial
                function.
            num_classes: The number of classes.
        """
        super().__init__(nnmodule, optimizer_partial, scheduler_partial)

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

    @typechecker
    def step(
        self: "BaseClassificationLitModule",
        batch: tuple[
            Float[Tensor, " batch_size *x_shape"],
            Float[Tensor, " batch_size"],
        ],
        stage: str,
    ) -> Tensor:
        """Step method common to all stages.

        Args:
            batch: Input data batch (images/sound/language/...).
            stage: Current stage (train/val/test).

        Returns:
            The loss.
        """
        # (BS x XS, BS) -> BS x XS, BS
        x, y = batch

        # BS x XS -> BS x NC
        logits = self(x)

        # BS x NC -> BS
        preds = torch.argmax(logits, dim=1)

        # Torchmetrics Accuracy not yet compatible with MyPy, see:
        # https://github.com/Lightning-AI/torchmetrics/blob/master/pyproject.toml
        accuracy = self.accuracy(preds, y)  # type: ignore[operator]

        self.log(f"{stage}/acc", accuracy)

        # (BS x NC, BS) -> 1
        return F.cross_entropy(logits, y)
