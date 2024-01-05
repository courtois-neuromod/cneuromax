"""MNIST classification task."""
from .datamodule import (
    MNISTClassificationDataModule,
    MNISTClassificationDataModuleConfig,
)
from .litmodule import MNISTClassificationLitModule

__all__ = [
    "MNISTClassificationDataModuleConfig",
    "MNISTClassificationDataModule",
    "MNISTClassificationLitModule",
]
