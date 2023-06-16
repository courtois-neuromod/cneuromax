"""MNIST Classification LitModule.

Abbreviations used in this module:

PyTorch ``nn.Module`` is short for ``torch.nn.Module``.

PyTorch ``Optimizer`` is short for ``torch.optim.Optimizer``.

PyTorch ``LRScheduler`` is short for
``torch.optim.lr_scheduler.LRScheduler``.

``Float`` is short for ``jaxtyping.Float``.
"""

from functools import partial

from beartype import beartype as typechecker
from jaxtyping import Float
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuroml.dl.base.classification import BaseClassificationLitModule


class MNISTClassificationLitModule(BaseClassificationLitModule):
    """MNIST Classification Model.

    Attributes:
        accuracy (``torchmetrics.Accuracy``): The accuracy metric.
        nnmodule (``nn.Module``): The PyTorch ``nn.Module`` instance.
        optimizer (``Optimizer``): The PyTorch ``Optimizer`` instance.
        scheduler (``LRScheduler``): The PyTorch ``LRScheduler``
            instance.
    """

    def __init__(
        self: "MNISTClassificationLitModule",
        nnmodule: nn.Module,
        optimizer_partial: partial[Optimizer],
        scheduler_partial: partial[LRScheduler],
    ) -> None:
        """Constructor.

        Calls parent constructor

        Args:
            nnmodule: A PyTorch ``nn.Module`` instance.
            optimizer_partial: A PyTorch ``Optimizer`` partial function.
            scheduler_partial: A PyTorch ``LRScheduler`` partial
                function.
        """
        super().__init__(
            nnmodule,
            optimizer_partial,
            scheduler_partial,
            num_classes=10,
        )

    @typechecker
    def forward(
        self: "MNISTClassificationLitModule",
        x: Float[Tensor, " batch_size 1 28 28"],
    ) -> Float[Tensor, " batch_size 10"]:
        """Forward method.

        Args:
            x: The batched MNIST images.

        Returns:
            The batched output logits.
        """
        out: Tensor = self.nnmodule(x)
        return out
