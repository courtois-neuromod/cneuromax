""":class:`ARLitModule` & its config."""
from abc import ABCMeta
from dataclasses import dataclass
from functools import partial
from typing import Annotated as An

import torch
import torch.nn.functional as f
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.classification import MulticlassAccuracy

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
)
from cneuromax.utils.beartype import ge, one_of


@dataclass
class ARLitModuleConfig:
    """Holds :class:`ARLitModule` config values."""


class ARLitModule(BaseLitModule):
    """``project`` :class:`.BaseLitModule`.

    Args:
        nnmodule: See :paramref:`~.BaseLitModule.nnmodule`.
        optimizer: See :paramref:`~.BaseLitModule.optimizer`.
        scheduler: See :paramref:`~.BaseLitModule.scheduler`.
    """

    def __init__(
        self: "ARLitModule",
        config: ARLitModuleConfig,
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        super().__init__(
            nnmodule=nnmodule,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.config = config

    def step(
        self: "ARLitModule",
        batch: tuple[Float[Tensor, " batch_size *x_shape"],],
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
        return torch.tensor(0.0)
