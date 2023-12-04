"""Entry point for Fitting with Deep Learning."""

import hydra
from omegaconf import DictConfig

from cneuromax.config import process_config
from cneuromax.fitting.deeplearning import store_deep_learning_fitting_configs
from cneuromax.fitting.deeplearning.config import (
    DeepLearningFittingHydraConfig,
    post_process_deep_learning_fitting_config,
    pre_process_deep_learning_fitting_config,
)
from cneuromax.fitting.deeplearning.fit import fit
from cneuromax.utils.wandb import login_wandb


@hydra.main(config_name="config", config_path=".", version_base=None)
def run(dict_config: DictConfig) -> None:
    """Processes the :mod:`hydra-core` config and fits w/ Deep Learning.

    Args:
        dict_config: The raw config object created by the
            :func:`hydra.main` decorator.
    """
    pre_process_deep_learning_fitting_config(dict_config)
    config = process_config(
        dict_config=dict_config,
        structured_config=DeepLearningFittingHydraConfig,
    )
    post_process_deep_learning_fitting_config(config)
    fit(config)


if __name__ == "__main__":
    store_deep_learning_fitting_configs()
    login_wandb()
    run()
