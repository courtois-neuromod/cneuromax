"""Deep net training + neural net evolution.

``config.yaml``:

.. highlight:: yaml
.. code-block:: yaml

    hydra:
        searchpath:
            - file://${oc.env:CNEUROMAX_PATH}/cneuromax/

    defaults:
        - deeplneuroevo_fitting
        - trainer: base
        - litmodule/scheduler: constant
        - litmodule/optimizer: adamw
        - logger: wandb
        - _self_
        - base_config
"""

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from cneuromax import store_project_configs
from cneuromax.config import process_config
from cneuromax.fitting import store_base_fitting_configs
from cneuromax.fitting.deeplneuroevo.config import (
    DeepLNeuroEvoFittingHydraConfig,
    post_process_deeplneuroevo_fitting_config,
    pre_process_deeplneuroevo_fitting_config,
)
from cneuromax.fitting.deeplneuroevo.fit import fit

__all__ = ["run", "store_configs"]


@hydra.main(config_name="config", config_path=".", version_base=None)
def run(config: DictConfig) -> None:
    """Processes the :mod:`hydra-core` config and fits w/ DL + NE.

    Args:
        config: See :paramref:`~.pre_process_base_config.config`.
    """
    pre_process_deeplneuroevo_fitting_config(config)
    config = process_config(
        config=config,
        structured_config_class=DeepLNeuroEvoFittingHydraConfig,
    )
    post_process_deeplneuroevo_fitting_config(config)
    fit(config)


def store_configs() -> None:
    """Stores :mod:`hydra-core` DL + NE fitting configs."""
    cs = ConfigStore.instance()
    store_project_configs(cs)
    store_base_fitting_configs(cs)
    cs.store(
        name="deeplneuroevo_fitting",
        node=DeepLNeuroEvoFittingHydraConfig,
    )
