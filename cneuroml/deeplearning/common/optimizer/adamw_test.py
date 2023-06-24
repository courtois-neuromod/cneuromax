"""Tests for base optimizer class."""

from hydra import compose, initialize

from cneuroml.deeplearning.common.optimizer.adam_test import BETAS, EPS
from cneuroml.deeplearning.common.optimizer.base_test import LR, WEIGHT_DECAY


def test() -> None:
    """."""
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="adamw")

    assert cfg._target_ == "torch.optim.AdamW"
    assert cfg._partial_
    assert cfg.lr == LR
    assert cfg.weight_decay == WEIGHT_DECAY
    assert cfg.betas == BETAS
    assert cfg.eps == EPS
