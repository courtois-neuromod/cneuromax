"""Tests for base optimizer class."""

from hydra import compose, initialize

from cneuroml.deeplearning.common.optimizer.base_test import LR, WEIGHT_DECAY

BETAS = [0.9, 0.999]
EPS = 1e-08


def test() -> None:
    """."""
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="adam")

    assert cfg._target_ == "torch.optim.Adam"
    assert cfg._partial_
    assert cfg.lr == LR
    assert cfg.weight_decay == WEIGHT_DECAY
    assert cfg.betas == BETAS
    assert cfg.eps == EPS
