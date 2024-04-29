"""."""

from hydra_zen import ZenStore

from cneuromax.projects.neuroevorl_control import (
    TaskRunner as NeuroevoRLControlTaskRunner,
)
from cneuromax.utils.hydra_zen import builds, fs_builds

from .agent import Agent, AgentConfig


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
            builds(Agent, config=fs_builds(AgentConfig)),
            name="agent",
            group="agent",
        )
