"""Fitting with Deep Learning & Neuroevolution.

``__main__.py`` (abridged):

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
        - hybrid_fitting
        - trainer: base
        - litmodule/scheduler: constant
        - litmodule/optimizer: adamw
        - logger: wandb
        - _self_
        - base_config
"""

from hydra.core.config_store import ConfigStore

from cneuromax import store_project_configs
from cneuromax.fitting import store_base_fitting_configs
from cneuromax.fitting.hybrid.config import (
    HybridFittingHydraConfig,
)


def store_hybrid_fitting_configs() -> None:
    """Stores :mod:`hydra-core` DL + NE fitting configs."""
    cs = ConfigStore.instance()
    store_project_configs(cs)
    store_base_fitting_configs(cs)
    cs.store(
        name="hybrid_fitting",
        node=HybridFittingHydraConfig,
    )
