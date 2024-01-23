"""Friends Finetuning task."""

from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner
from cneuromax.utils.hydra_zen import fs_builds

from .friends_datamodule import (
    FriendsDataModule,
    FriendsDataModuleConfig,
)
from .friends_finetune_model import (
    FriendsFinetuningModel,
)

__all__ = [
    "TaskRunner",
    "FriendsDataModule",
    "FriendsDataModuleConfig",
    "FriendsFinetuningModel",
]


class TaskRunner(DeepLearningTaskRunner):
    """MNIST classification ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra-core` MNIST classification configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store=store)
        store(
            fs_builds(
                FriendsDataModule,
                config=FriendsDataModuleConfig(),
            ),
            name="friends_language_encoder",
            group="datamodule",
        )
        store(
            fs_builds(FriendsFinetuningModel),
            name="friends_language_encoder",
            group="litmodule",
        )
