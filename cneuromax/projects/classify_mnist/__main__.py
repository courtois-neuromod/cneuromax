"""MNIST classification ``task`` runner."""
from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner
from cneuromax.utils.zen import fs_builds

from .datamodule import (
    MNISTClassificationDataModule,
    MNISTClassificationDataModuleConfig,
)
from .litmodule import MNISTClassificationLitModule


class TaskRunner(DeepLearningTaskRunner):
    """MNIST classification ``task`` runner."""

    @staticmethod
    def store_configs(store: ZenStore) -> None:
        """Stores :mod:`hydra-core` MNIST classification configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        store(
            fs_builds(
                MNISTClassificationDataModule,
                config=MNISTClassificationDataModuleConfig(),
            ),
            name="classify_mnist",
            group="datamodule",
        )
        store(
            fs_builds(
                MNISTClassificationLitModule,
                config=MNISTClassificationDataModuleConfig(),
            ),
            name="classify_mnist",
            group="litmodule",
        )


if __name__ == "__main__":
    TaskRunner.store_configs_and_run_task()
