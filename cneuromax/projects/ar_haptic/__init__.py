"""Haptic Autoregression ``project``."""
from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner
from cneuromax.utils.hydra_zen import fs_builds

from .datamodule import ARDataModule, ARDataModuleConfig
from .litmodule import ARLitModule, ARLitModuleConfig

__all__ = [
    "TaskRunner",
    "ARDataModule",
    "ARDataModuleConfig",
    "ARLitModule",
    "ARLitModuleConfig",
]


class TaskRunner(DeepLearningTaskRunner):
    """Haptic autoregression ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra-core` Haptic autoregression configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store=store)
        store(
            fs_builds(ARDataModule, config=ARDataModuleConfig()),
            name="ar_haptic",
            group="datamodule",
        )
        store(
            fs_builds(ARLitModule, config=ARLitModuleConfig()),
            name="classify_mnist",
            group="ar_haptic",
        )
