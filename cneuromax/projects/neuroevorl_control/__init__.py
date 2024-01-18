"""Reinforcement Neuroevolution on Control environments ``project``."""
from hydra_zen import ZenStore

from cneuromax.fitting.neuroevolution.runner import NeuroevolutionTaskRunner
from cneuromax.utils.hydra_zen import builds, fs_builds

from .agent import GymAgent, GymAgentConfig
from .space import GymReinforcementSpace, GymReinforcementSpaceConfig

__all__ = [
    "TaskRunner",
    "GymAgent",
    "GymAgentConfig",
    "GymReinforcementSpace",
    "GymReinforcementSpaceConfig",
]


class TaskRunner(NeuroevolutionTaskRunner):
    """``project`` ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra-core` ``project`` configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store)
        store(
            fs_builds(
                GymReinforcementSpace,
                config=fs_builds(GymReinforcementSpaceConfig),
            ),
            name="rl_control_nevo",
            group="space",
        )
        store(
            builds(
                GymAgent,
                config=fs_builds(GymAgentConfig),
            ),
            name="rl_control_nevo",
            group="agent",
        )
