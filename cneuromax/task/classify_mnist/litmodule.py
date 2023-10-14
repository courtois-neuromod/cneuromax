"""Lightning Module for MNIST classification."""

from functools import partial

from jaxtyping import Float
from torch import Tensor, nn
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

    def forward(
        self: "MNISTClassificationLitModule",
        x: Float[Tensor, " BS 1 28 28"],
    ) -> Float[Tensor, " BS 10"]:
        """Simple pass through the PyTorch ``nn.Module``.

        Args:
            x: The batched MNIST images.

        Returns:
            The batched output logits.
        """
        out: Float[Tensor, " BS 10"] = self.nnmodule(x)
        return out
