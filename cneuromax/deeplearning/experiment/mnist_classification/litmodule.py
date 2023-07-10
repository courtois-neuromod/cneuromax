"""."""

from dataclasses import dataclass
from functools import partial

from beartype import beartype as typechecker
from jaxtyping import Float
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.deeplearning.common.litmodule.classification import (
    BaseClassificationLitModule,
    BaseClassificationLitModuleConfig,
)


@dataclass
class MNISTClassificationLitModuleConfig(BaseClassificationLitModuleConfig):
    """.

    Attributes:
        num_classes: .
    """

    num_classes: int = 10


class MNISTClassificationLitModule(BaseClassificationLitModule):
    """.

    Attributes:
        accuracy (``torchmetrics.Accuracy``): The accuracy metric.
        nnmodule (``nn.Module``): .
        optimizer (``Optimizer``): .
        scheduler (``LRScheduler``): .
    """

    def __init__(
        self: "MNISTClassificationLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        lrscheduler: partial[LRScheduler],
    ) -> None:
        """Calls parent constructor.

        Args:
            nnmodule: A PyTorch ``nn.Module`` instance.
            optimizer: A PyTorch ``Optimizer`` partial function.
            lrscheduler: A PyTorch ``LRScheduler`` partial
                function.
        """
        super().__init__(
            nnmodule,
            optimizer,
            lrscheduler,
            config=MNISTClassificationLitModuleConfig(),
        )

    @typechecker
    def forward(
        self: "MNISTClassificationLitModule",
        x: Float[Tensor, " batch_size 1 28 28"],
    ) -> Float[Tensor, " batch_size 10"]:
        """Simple pass through the PyTorch ``nn.Module``.

        Args:
            x: The batched MNIST images.

        Returns:
            The batched output logits.
        """
        out: Tensor = self.nnmodule(x)
        return out
