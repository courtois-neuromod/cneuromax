"""Tests for base optimizer class."""

from hydra import compose, initialize

LAST_EPOCH = -1


def test() -> None:
    """."""
    with initialize(version_base=None, config_path="."):
        cfg = compose(
            config_name="base",
            overrides=["_target_=transformers.get_constant_schedule"],
        )

    assert cfg._partial_
    assert cfg.last_epoch == LAST_EPOCH
