"""."""

from cneuromax.deeplearning.common.optimizer import SGDConfig
from cneuromax.deeplearning.common.optimizer.base_test import LR, WEIGHT_DECAY

MOMENTUM = 0


def test() -> None:
    """."""
    config = SGDConfig()
    assert config._target_ == "torch.optim.SGD"
    assert config.momentum == MOMENTUM
    assert config._partial_ is True
    assert config.lr == LR
    assert config.weight_decay == WEIGHT_DECAY
