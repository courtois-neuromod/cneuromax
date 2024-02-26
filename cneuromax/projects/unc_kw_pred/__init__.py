""":mod:`.unc_kw_pred` project."""

from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner

from .datamodule import store_configs as store_datamodule_configs
from .litmodule import store_configs as store_litmodule_configs


class TaskRunner(DeepLearningTaskRunner):
    """:mod:`.kw_pred` ``project`` ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra-core` ``project`` configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store=store)
        store_datamodule_configs(store=store)
        store_litmodule_configs(store=store)
