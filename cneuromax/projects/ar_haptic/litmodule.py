""":class:`ARLitModule` + its config."""
from dataclasses import dataclass
from functools import partial

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
)


@dataclass
class ARLitModuleConfig:
    """Configuration for :class:`ARHapticLitModule`."""


class ARLitModule(BaseLitModule):
    """Autore Classification :mod:`lightning` Module.

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
