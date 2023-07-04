"""Tests for base optimizer class."""

from hydra import compose, initialize

from cneuromax.deeplearning.common.lrscheduler.base_test import LAST_EPOCH


def test() -> None:
    """."""
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="none")

    assert cfg._target_ == "transformers.get_constant_schedule"
    assert cfg._partial_
    assert cfg.last_epoch == LAST_EPOCH
