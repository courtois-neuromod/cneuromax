"""Control task neuroevolution ``project``."""
from hydra_zen import ZenStore, make_config

from cneuromax.fitting.neuroevolution.runner import NeuroevolutionTaskRunner
from cneuromax.utils.hydra_zen import fs_builds

from .agent import GymAgent, GymAgentConfig
from .space import GymReinforcementSpace, GymReinforcementSpaceConfig

__all__ = [
    "TaskRunner",
]


class TaskRunner(NeuroevolutionTaskRunner):
    """MNIST classification ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra-core` MNIST classification configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store)
        store(
            fs_builds(
                GymReinforcementSpace,
                config=GymReinforcementSpaceConfig(),
            ),
            name="rl_control_nevo",
            group="space",
        )
        store(
            fs_builds(
                GymAgent,
                config=GymAgentConfig(),
            ),
            name="rl_control_nevo",
            group="agent",
        )
