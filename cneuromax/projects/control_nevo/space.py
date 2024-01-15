""":class:`GymReinforcementSpace` & its config."""
from dataclasses import dataclass
from typing import Any

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

    env_name: str = "Acrobot-v1"


class GymReinforcementSpace(BaseReinforcementSpace):
    """:class:`.BaseSpace` for reinforcement w/ :mod:`gymnasium`."""

    def __init__(
        self: "GymReinforcementSpace",
        config: GymReinforcementSpaceConfig,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        env = GymEnv(env_name=config.env_name)
        super().__init__(config=config, env=env, args=args, kwargs=kwargs)
