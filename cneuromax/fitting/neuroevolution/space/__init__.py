"""Neuroevolution Spaces."""

from cneuromax.fitting.neuroevolution.space.base import (
    BaseSpace,
    BaseSpaceConfig,
)
from cneuromax.fitting.neuroevolution.space.reinforcement import (
    BaseReinforcementSpace,
)

__all__ = [
    "BaseReinforcementSpace",
    "BaseSpace",
    "BaseSpaceConfig",
]
