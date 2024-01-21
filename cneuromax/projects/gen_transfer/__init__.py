"""Neuroevolution generational transfer ``project``."""
from hydra_zen import ZenStore

from cneuromax.projects.neuroevorl_control import (
    TaskRunner as NeuroevoRLControlTaskRunner,
)
from cneuromax.utils.hydra_zen import builds, fs_builds

from .agent import TransferAgent, TransferAgentConfig

__all__ = ["TaskRunner", "TransferAgent", "TransferAgentConfig"]


class TaskRunner(NeuroevoRLControlTaskRunner):
    """``gen_transfer project`` ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra-core` MNIST classification configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store)
        store(
            builds(TransferAgent, config=fs_builds(TransferAgentConfig)),
            name="gen_transfer",
            group="agent",
        )
