""":class:`MNISTClassificationLitModule`."""
from functools import partial

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.fitting.deeplearning.litmodule.classification import (
    BaseClassificationLitModule,
    BaseClassificationLitModuleConfig,
)


class MNISTClassificationLitModule(BaseClassificationLitModule):
    """MNIST Classification :mod:`lightning` Module.

    Args:
        nnmodule: See :paramref:`~.BaseLitModule.nnmodule`.
        optimizer: See :paramref:`~.BaseLitModule.optimizer`.
        scheduler: See :paramref:`~.BaseLitModule.scheduler`.
    """

    def __init__(
        self: "MNISTClassificationLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        super().__init__(
            BaseClassificationLitModuleConfig(num_classes=10),
            nnmodule,
            optimizer,
            scheduler,
        )
