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
        job:
            chdir: True
        searchpath:
            - file://${oc.env:CNEUROMAX_PATH}/cneuromax/
        run:
            dir: ${run_dir}/
        sweep:
            dir: ${run_dir}/

        defaults:
            - neuroevolution_fitting
            - _self_
            - task: null
"""

from hydra.core.config_store import ConfigStore

from cneuromax import store_task_configs
from cneuromax.fitting import store_base_fitting_configs
from cneuromax.fitting.neuroevolution.config import (
    NeuroevolutionFittingHydraConfig,
)

__all__ = ["store_neuroevolution_fitting_configs"]


def store_neuroevolution_fitting_configs() -> None:
    """Stores :mod:`hydra-core` Neuroevolution fitting configs."""
    cs = ConfigStore.instance()
    store_task_configs(cs)
    store_base_fitting_configs(cs)
    cs.store(
        name="neuroevolution_fitting",
        node=NeuroevolutionFittingHydraConfig,
    )
