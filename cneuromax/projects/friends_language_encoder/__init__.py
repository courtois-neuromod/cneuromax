"""Friends language finetuning ``project``."""

from hydra_zen import ZenStore
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM

from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner
from cneuromax.utils.hydra_zen import fs_builds

from .datamodule import (
    FriendsDataModule,
    FriendsDataModuleConfig,
)
from .litmodule import (
    FriendsFinetuningModel,
    FriendsLitModuleConfig,
)

__all__ = [
    "TaskRunner",
    "FriendsDataModule",
    "FriendsDataModuleConfig",
    "FriendsFinetuningModel",
    "FriendsLitModuleConfig",
]


class TaskRunner(DeepLearningTaskRunner):
    """``project`` ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra-core` ``project`` configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store=store)
        store(name="model_name")
        store(
            fs_builds(
                FriendsDataModule,
                config=FriendsDataModuleConfig(),
            ),
            name="friends_language_encoder",
            group="datamodule",
        )
        store(
            fs_builds(
                FriendsFinetuningModel,
                config=FriendsLitModuleConfig(),
            ),
            name="friends_language_encoder",
            group="litmodule",
        )
        store(
            fs_builds(
                AutoModelForCausalLM.from_pretrained,
                pretrained_model_name_or_path="${model_name}",
            ),
            name="friends_language_encoder",
            group="litmodule/nnmodule",
        )
