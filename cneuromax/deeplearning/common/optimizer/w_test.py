"""."""

from cneuromax.deeplearning.common.optimizer import AdamWConfig
from cneuromax.deeplearning.common.optimizer.adam_test import BETAS, EPS
from cneuromax.deeplearning.common.optimizer.base_test import LR, WEIGHT_DECAY


def test() -> None:
    """."""
    config = AdamWConfig()
    assert config._target_ == "torch.optim.AdamW"
    assert config.betas == BETAS
    assert config.eps == EPS
    assert config._partial_ is True
    assert config.lr == LR
    assert config.weight_decay == WEIGHT_DECAY
