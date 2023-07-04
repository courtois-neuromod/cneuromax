"""."""

from hydra.core.config_store import ConfigStore

from cneuromax.deeplearning.common.optimizer.adam import AdamConfig
from cneuromax.deeplearning.common.optimizer.adamw import AdamWConfig
from cneuromax.deeplearning.common.optimizer.sgd import SGDConfig

__all__ = ["AdamConfig", "AdamWConfig", "SGDConfig"]


def store(cs: ConfigStore) -> None:
    """."""
    cs.store(group="litmodule.optimizer", name="adam", node=AdamConfig)
    cs.store(group="litmodule.optimizer", name="adamw", node=AdamWConfig)
    cs.store(group="litmodule.optimizer", name="sgd", node=SGDConfig)
