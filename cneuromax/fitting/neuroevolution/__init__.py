"""Neural net evolution.

``config.yaml``:

.. highlight:: yaml
.. code-block:: yaml

    hydra:
        searchpath:
            - file://${oc.env:CNEUROMAX_PATH}/cneuromax/

    defaults:
        - neuroevolution_fitting
        - logger: wandb_simexp
        - _self_
        - base_config
"""

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from cneuromax import store_project_configs
from cneuromax.config import process_config
from cneuromax.fitting import store_base_fitting_configs
from cneuromax.fitting.neuroevolution.config import (
    NeuroevolutionFittingHydraConfig,
    post_process_neuroevolution_fitting_config,
    pre_process_neuroevolution_fitting_config,
)
from cneuromax.fitting.neuroevolution.fit import fit
from cneuromax.utils.wandb import store_logger_configs

__all__ = ["run", "store_configs"]


@hydra.main(config_name="config", config_path=".", version_base=None)
def run(config: DictConfig) -> None:
    """Processes the :mod:`hydra-core` config & fits w/ Neuroevolution.

    Args:
        config: See :paramref:`~.pre_process_base_config.config`.
    """
    pre_process_neuroevolution_fitting_config(config)
    config = process_config(
        config=config,
        structured_config_class=NeuroevolutionFittingHydraConfig,
    )
    post_process_neuroevolution_fitting_config(config)
    fit(config)


def store_configs() -> None:
    """Stores :mod:`hydra-core` Neuroevolution fitting configs."""
    cs = ConfigStore.instance()
    store_project_configs(cs)
    store_base_fitting_configs(cs)
    store_logger_configs(cs, wandb.init)
    cs.store(
        name="neuroevolution_fitting",
        node=NeuroevolutionFittingHydraConfig,
    )
