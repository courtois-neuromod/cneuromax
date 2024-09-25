""":class:`.GymReinforcementSpace` & its config."""

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
    """``project`` :class:`.BaseReinforcementSpace`.

    Args:
        config: See :class:`GymReinforcementSpaceConfig`.
    """

    def __init__(
        self: "GymReinforcementSpace",
        config: GymReinforcementSpaceConfig,
    ) -> None:
        super().__init__(
            config=config,
            env=GymEnv(env_name=config.env_name),
        )
