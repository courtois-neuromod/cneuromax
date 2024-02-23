"""MNIST classification ``project``."""

from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner
from cneuromax.utils.hydra_zen import fs_builds

from .datamodule import MNISTDataModule, MNISTDataModuleConfig
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
        """Stores :mod:`hydra-core` ``project`` configs.

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
            fs_builds(MNISTClassificationLitModule),
            name="classify_mnist",
            group="litmodule",
        )
