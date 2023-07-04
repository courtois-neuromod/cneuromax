"""."""

from omegaconf import MISSING

from cneuromax.deeplearning.common.optimizer.base import BaseOptimizerConfig

LR = 1e-3
WEIGHT_DECAY = 1e-2


def test() -> None:
    """."""
    config = BaseOptimizerConfig()
    assert config._target_ == MISSING
    assert config._partial_ is True
    assert config.lr == LR
    assert config.weight_decay == WEIGHT_DECAY
