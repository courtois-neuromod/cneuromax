"""MNIST classification ``project``."""

from hydra_zen import ZenStore
from torch import nn

from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner
from cneuromax.utils.hydra_zen import fs_builds

from .datamodules.replay_datamodule import (
    ReplayDataModule,
    ReplayDataModuleConfig,
)
from .videogpt.vqvae import VQVAE, VQVAEConfig

__all__ = [
    "TaskRunner",
    "ReplayDataModule",
    "ReplayDataModuleConfig",
    "VQVAE",
    "VQVAEConfig",
]


class DummyNNModule(nn.Module):
    """Dummy NN module."""


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
            fs_builds(ReplayDataModule, config=ReplayDataModuleConfig()),
            name="replay_datamodule",
            group="datamodule",
        )
        store(
            fs_builds(
                VQVAE,
                config=VQVAEConfig(),
            ),
            name="vqvae",
            group="litmodule",
        )
        store(
            fs_builds(
                DummyNNModule,
            ),
            name="dummy_nnmodule",
            group="litmodule/nnmodule",
        )
