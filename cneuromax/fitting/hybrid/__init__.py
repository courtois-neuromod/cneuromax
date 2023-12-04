"""Fitting w/ Hybrid DL & NE (+ :mod:`hydra-core` config storing)."""

from hydra.core.config_store import ConfigStore

from cneuromax import store_task_configs
from cneuromax.fitting import store_base_fitting_configs
from cneuromax.fitting.hybrid.config import (
    HybridFittingHydraConfig,
)


def store_hybrid_fitting_configs() -> None:
    """Stores :mod:`hydra-core` Hybrid DL + NE fitting configs."""
    cs = ConfigStore.instance()
    store_task_configs(cs)
    store_base_fitting_configs(cs)
    cs.store(name="hybrid_fitting", node=HybridFittingHydraConfig)
