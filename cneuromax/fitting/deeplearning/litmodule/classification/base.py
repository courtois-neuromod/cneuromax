"""Base classification LitModule config & class."""

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

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.annotations import ge, one_of


@dataclass
class BaseClassificationLitModuleConfig:
    """Config for :class:`BaseClassificationLitModule` instances.

    Attributes:
        num_classes: Number of classes to classify.
    """

    num_classes: An[int, ge(2)]


class BaseClassificationLitModule(BaseLitModule, metaclass=ABCMeta):
    """Base Classification :class:`~lightning.pytorch.LightningModule`.

    Attributes:
        accuracy\
            (:class:`~torchmetrics.classification.MulticlassAccuracy`):\
            The accuracy metric.
    """

    def __init__(
        self: "BaseClassificationLitModule",
        config: BaseClassificationLitModuleConfig,
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        """Calls parent constructor & initializes instance attributes.

        Args:
            config: This instance's configuration, see\
                :class:`BaseClassificationLitModuleConfig`.
            nnmodule: See\
                :paramref:`~cneuromax.fitting.deeplearning.litmodule.base.BaseLitModule.nnmodule`.
            optimizer: See\
                :paramref:`~cneuromax.fitting.deeplearning.litmodule.base.BaseLitModule.optimizer`.
            scheduler: See\
                :paramref:`~cneuromax.fitting.deeplearning.litmodule.base.BaseLitModule.scheduler`.
        """
        super().__init__(nnmodule, optimizer, scheduler)
        self.accuracy: MulticlassAccuracy = MulticlassAccuracy(
            num_classes=config.num_classes,
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
            batch: The input data batch.
            stage: The current stage.

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
