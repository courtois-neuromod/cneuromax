"""Entry point for Deep Learning fitting."""

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from cneuromax import login_wandb, process_config, store_task_configs
from cneuromax.fitting import store_configs as store_base_fitting_configs
from cneuromax.fitting.deeplearning import (
    store_configs as store_deep_learning_configs,
)
from cneuromax.fitting.deeplearning.fit import (
    DeepLearningFittingHydraConfig,
    fit,
)


def store_configs() -> None:
    """Store configs for the Deep Learning module."""
    cs = ConfigStore.instance()
    store_base_fitting_configs(cs)
    store_deep_learning_configs(cs)
    store_task_configs(cs)


@hydra.main(config_name="config", config_path=".", version_base=None)
def run(dict_config: DictConfig) -> None:
    """Processes the config and fits with Deep Learning.

    Args:
        dict_config: The raw config object created by the
            :func:`hydra.main` decorator.
    """
    config = process_config(
        dict_config=dict_config,
        config_class=DeepLearningFittingHydraConfig,
    )
    fit(config)


if __name__ == "__main__":
    store_configs()
    login_wandb()
    run()
