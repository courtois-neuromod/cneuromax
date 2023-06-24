"""Tests for base optimizer class."""

from hydra import compose, initialize

from cneuroml.deeplearning.common.optimizer.base_test import LR, WEIGHT_DECAY

MOMENTUM = 0


def test() -> None:
    """."""
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="sgd")

    assert cfg._target_ == "torch.optim.SGD"
    assert cfg._partial_
    assert cfg.lr == LR
    assert cfg.weight_decay == WEIGHT_DECAY
    assert cfg.momentum == MOMENTUM
