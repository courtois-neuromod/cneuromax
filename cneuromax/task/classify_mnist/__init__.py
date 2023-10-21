"""MNIST classification task."""

from hydra.core.config_store import ConfigStore

from cneuromax.task.classify_mnist.datamodule import (
    MNISTClassificationDataModule,
    MNISTClassificationDataModuleConfig,
)
from cneuromax.task.classify_mnist.litmodule import (
    MNISTClassificationLitModule,
)
from cneuromax.utils.hydra import fs_builds

__all__ = [
    "MNISTClassificationDataModule",
    "MNISTClassificationDataModuleConfig",
    "MNISTClassificationLitModule",
]


def store_configs(cs: ConfigStore) -> None:
    """Stores the MNIST classification configs.

    Args:
        cs: .
    """
    cs.store(
        group="datamodule",
        name="classify_mnist",
        node=fs_builds(
            MNISTClassificationDataModule,
            config=MNISTClassificationDataModuleConfig(),
        ),
    )
    cs.store(
        group="litmodule",
        name="classify_mnist",
        node=fs_builds(MNISTClassificationLitModule),
    )
