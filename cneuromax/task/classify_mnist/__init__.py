"""MNIST classification task."""

from hydra.core.config_store import ConfigStore
from hydra_zen import builds

from cneuromax.task.classify_mnist.datamodule import (
    MNISTClassificationDataModule,
    MNISTClassificationDataModuleConfig,
)
from cneuromax.task.classify_mnist.litmodule import (
    MNISTClassificationLitModule,
)

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
        node=builds(
            MNISTClassificationDataModule,
            config=MNISTClassificationDataModuleConfig(),
        ),
    )
    cs.store(
        group="litmodule",
        name="classify_mnist",
        node=builds(
            MNISTClassificationLitModule,
            populate_full_signature=True,
        ),
    )
