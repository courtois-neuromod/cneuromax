"""Neuroevolution Spaces."""

from cneuromax.fitting.neuroevolution.space.base import BaseSpace
from cneuromax.fitting.neuroevolution.space.imitation import (
    BaseImitationSpace,
    BaseImitationTarget,
)
from cneuromax.fitting.neuroevolution.space.reinforcement import (
    BaseReinforcementSpace,
)

__all__ = [
    "BaseBatchImitationSpace",
    "BaseBatchImitationTarget",
    "BaseBatchReinforcementSpace",
    "BaseImitationSpace",
    "BaseImitationTarget",
    "BaseReinforcementSpace",
    "BaseSpace",
]
