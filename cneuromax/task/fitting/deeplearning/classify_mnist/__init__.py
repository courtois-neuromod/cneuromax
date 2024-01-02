"""MNIST classification task."""

from hydra.core.config_store import ConfigStore

from cneuromax.utils.hydra import fs_builds

from .datamodule import (
    MNISTClassificationDataModule,
    MNISTClassificationDataModuleConfig,
)
from .litmodule import MNISTClassificationLitModule

__all__ = [
    "store_configs",
    "MNISTClassificationDataModuleConfig",
    "MNISTClassificationDataModule",
    "MNISTClassificationLitModule",
]


def store_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` MNIST classification configs.

    Args:
        cs: See :paramref:`~.store_project_configs.cs`.
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
