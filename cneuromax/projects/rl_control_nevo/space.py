""":class:`GymReinforcementSpace` & its config."""
from dataclasses import dataclass

from omegaconf import MISSING
from torchrl.envs.libs.gym import GymEnv

from cneuromax.fitting.neuroevolution.space import (
    BaseReinforcementSpace,
    BaseSpaceConfig,
)


@dataclass
class GymReinforcementSpaceConfig(BaseSpaceConfig):
    """Holds :class:`GymReinforcementSpace` config values.

    Args:
        env_name: The name of the :mod:`gymnasium` environment.
    """

    env_name: str = MISSING


class GymReinforcementSpace(BaseReinforcementSpace):
    """:class:`.BaseSpace` for reinforcement on :mod:`gymnasium`.

    Args:
        config: See :class:`GymReinforcementSpaceConfig`.
        num_pops: See :paramref:`~.BaseSpace.num_pops`.
        evaluates_on_gpu: See :paramref:`~.BaseSpace.evaluates_on_gpu`.
    """

    def __init__(
        self: "GymReinforcementSpace",
        config: GymReinforcementSpaceConfig,
        num_pops: int,
        *,
        evaluates_on_gpu: bool,
    ) -> None:
        super().__init__(
            config=config,
            env=GymEnv(env_name=config.env_name),
            num_pops=num_pops,
            evaluates_on_gpu=evaluates_on_gpu,
        )
