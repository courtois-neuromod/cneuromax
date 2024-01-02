"""Fitting with Neuroevolution.

``__main__.py`` abridged code:

.. highlight:: python
.. code-block:: python

    @hydra.main(config_name="config", config_path=".")
    def run(config: DictConfig) -> None:
        config = process(config)
        fit(config)

    if __name__ == "__main__":
        run()

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

import wandb
from hydra.core.config_store import ConfigStore

from cneuromax import store_project_configs
from cneuromax.fitting import store_base_fitting_configs
from cneuromax.fitting.neuroevolution.config import (
    NeuroevolutionFittingHydraConfig,
)
from cneuromax.utils.wandb import store_logger_configs

__all__ = ["store_neuroevolution_fitting_configs"]


def store_neuroevolution_fitting_configs() -> None:
    """Stores :mod:`hydra-core` Neuroevolution fitting configs."""
    cs = ConfigStore.instance()
    store_project_configs(cs)
    store_base_fitting_configs(cs)
    store_logger_configs(cs, wandb.init)
    cs.store(
        name="neuroevolution_fitting",
        node=NeuroevolutionFittingHydraConfig,
    )
