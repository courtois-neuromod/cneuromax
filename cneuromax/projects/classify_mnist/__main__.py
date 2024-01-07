""":class:`TaskRunner`."""
from hydra_zen import ZenStore, make_config

from cneuromax.fitting.deeplearning.nnmodule.mlp import store_mlp_config
from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner
from cneuromax.utils.hydra_zen import fs_builds

from .datamodule import (
    MNISTClassificationDataModule,
    MNISTClassificationDataModuleConfig,
)
from .litmodule import MNISTClassificationLitModule


class TaskRunner(DeepLearningTaskRunner):
    """MNIST classification ``task`` runner."""

    # @classmethod
    # def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
    #     """Stores :mod:`hydra-core` MNIST classification configs.

    #     Args:
    #         store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    #     """
    #     super().store_configs(store)
    #     store(
    #         make_config(
    #             MNISTClassificationDataModule,
    #             config=MNISTClassificationDataModuleConfig(),
    #         ),
    #         name="classify_mnist",
    #         group="datamodule",
    #     )
    #     store(
    #         fs_builds(MNISTClassificationLitModule),
    #         name="classify_mnist",
    #         group="litmodule",
    #     )
    #     store_mlp_config(store)


if __name__ == "__main__":
    TaskRunner.store_configs_and_run_task()
