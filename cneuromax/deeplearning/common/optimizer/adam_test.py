"""."""

from cneuromax.deeplearning.common.optimizer import AdamConfig
from cneuromax.deeplearning.common.optimizer.base_test import LR, WEIGHT_DECAY

BETAS = [0.9, 0.999]
EPS = 1e-08


def test() -> None:
    """."""
    config = AdamConfig()
    assert config._target_ == "torch.optim.Adam"
    assert config.betas == BETAS
    assert config.eps == EPS
    assert config._partial_ is True
    assert config.lr == LR
    assert config.weight_decay == WEIGHT_DECAY
