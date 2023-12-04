"""Fitting with Neuroevolution (+ :mod:`hydra-core` config storing)."""

from hydra.core.config_store import ConfigStore

from cneuromax import store_task_configs
from cneuromax.fitting import store_base_fitting_configs
from cneuromax.fitting.neuroevolution.config import (
    NeuroevolutionFittingHydraConfig,
)


def store_neuroevolution_fitting_configs() -> None:
    """Stores :mod:`hydra-core` Neuroevolution fitting configs."""
    cs = ConfigStore.instance()
    store_task_configs(cs)
    store_base_fitting_configs(cs)
    cs.store(
        name="neuroevolution_fitting",
        node=NeuroevolutionFittingHydraConfig,
    )
