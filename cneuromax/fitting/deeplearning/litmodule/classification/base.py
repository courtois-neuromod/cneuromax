""":class:`BaseClassificationLitModule` & its config dataclass."""
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


@dataclass(frozen=True)
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
    """

    def __init__(
        self: "BaseClassificationLitModule",
        config: BaseClassificationLitModuleConfig,
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
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
        return f.cross_entropy(input=logits, target=y)
