"""Neuroevolution Spaces."""

from cneuromax.fitting.neuroevolution.space.base import BaseSpace
from cneuromax.fitting.neuroevolution.space.batch_imitation import (
    BaseBatchImitationSpace,
    BaseBatchImitationTarget,
)
from cneuromax.fitting.neuroevolution.space.batch_reinforcement import (
    BaseBatchReinforcementSpace,
)
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
