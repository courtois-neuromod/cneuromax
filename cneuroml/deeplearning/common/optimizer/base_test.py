"""Tests for base optimizer class."""

from hydra import compose, initialize

LR = 0.001
WEIGHT_DECAY = 0.01


def test() -> None:
    """."""
    with initialize(version_base=None, config_path="."):
        cfg = compose(
            config_name="base",
            overrides=["_target_=torch.optim.SGD"],
        )

    assert cfg._partial_
    assert cfg.lr == LR
    assert cfg.weight_decay == WEIGHT_DECAY
