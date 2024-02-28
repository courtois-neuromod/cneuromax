"""MNIST classification ``project``."""

from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.litmodule.classification import (
    BaseClassificationLitModuleConfig,
)
from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner
from cneuromax.projects.classify_mnist.datamodule import (
    MNISTDataModule,
    MNISTDataModuleConfig,
)
from cneuromax.utils.hydra_zen import fs_builds

from .litmodule import MNISTClassificationLitModule

__all__ = [
    "TaskRunner",
    "MNISTDataModuleConfig",
    "MNISTDataModule",
    "MNISTClassificationLitModule",
]


class TaskRunner(DeepLearningTaskRunner):
    """:mod:`.classify_mnist` ``project`` ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra` ``project`` configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store=store)
        store(
            fs_builds(MNISTDataModule, config=MNISTDataModuleConfig()),
            name="mnist",
            group="datamodule",
        )
        store(
            fs_builds(
                MNISTClassificationLitModule,
                config=BaseClassificationLitModuleConfig(num_classes=10),
            ),
            name="classify_mnist",
            group="litmodule",
        )
