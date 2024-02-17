"""MNIST generation ``project``."""

from denoising_diffusion_pytorch import Unet
from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner
from cneuromax.projects.classify_mnist import (
    MNISTDataModule,
    MNISTDataModuleConfig,
)
from cneuromax.utils.hydra_zen import fs_builds

from .litmodule import MNISTGenerationLitModule

__all__ = [
    "TaskRunner",
    "MNISTDataModuleConfig",
    "MNISTDataModule",
    "MNISTGenerationLitModule",
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
            fs_builds(Unet, dim=64, dim_mults=(1, 2), channels=1),
            name="unet",
            group="litmodule/nnmodule",
        )
        store(
            fs_builds(MNISTGenerationLitModule),
            name="gen_mnist",
            group="litmodule",
        )
