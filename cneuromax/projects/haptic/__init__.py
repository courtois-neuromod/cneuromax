"""Haptic ``project``."""
from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner

from .klk_wav import store_configs as store_klk_wav_configs
from .one_klk_wav import store_configs as store_one_klk_wav_configs


class TaskRunner(DeepLearningTaskRunner):
    """``project`` ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra-core` ``project`` configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store=store)
        store_klk_wav_configs(store=store)
        store_one_klk_wav_configs(store=store)
