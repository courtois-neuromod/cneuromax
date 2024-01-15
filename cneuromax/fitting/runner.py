""":class:`FittingTaskRunner`."""
from hydra_zen import ZenStore

from cneuromax.fitting.store import store_launcher_configs
from cneuromax.runner import BaseTaskRunner


class FittingTaskRunner(BaseTaskRunner):
    """Fitting ``task`` runner.

    Attributes:
        subtask_config: See :attr:`~.BaseTaskRunner.subtask_config`.
    """

    @classmethod
    def store_configs(cls: type["FittingTaskRunner"], store: ZenStore) -> None:
        """Stores structured configs.

        .. warning::

            Make sure to call this method if you are overriding it.

        Args:
            store: See :meth:`~.BaseTaskRunner.store_configs`.
        """
        super().store_configs(store)
        store_launcher_configs(store)
