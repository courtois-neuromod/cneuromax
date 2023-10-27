"""Lightning Module for MNIST classification."""


from functools import partial

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.fitting.deeplearning.litmodule.classification import (
    BaseClassificationLitModule,
)


class MNISTClassificationLitModule(BaseClassificationLitModule):
    """MNIST classification Lightning Module."""

    def __init__(
        self: "MNISTClassificationLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        """Calls parent constructor.

        Args:
            nnmodule: .
            optimizer: .
            scheduler: .
        """
        super().__init__(nnmodule, optimizer, scheduler, num_classes=10)
