"""Entry point for Fitting with Neuroevolution."""

import hydra
from omegaconf import DictConfig

from cneuromax.config import process_config
from cneuromax.fitting.neuroevolution import (
    store_neuroevolution_fitting_configs,
)
from cneuromax.fitting.neuroevolution.config import (
    NeuroevolutionFittingHydraConfig,
    post_process_neuroevolution_fitting_config,
    pre_process_neuroevolution_fitting_config,
)
from cneuromax.fitting.neuroevolution.fit import fit
from cneuromax.utils.wandb import login_wandb


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


if __name__ == "__main__":
    store_neuroevolution_fitting_configs()
    login_wandb()
    run()
