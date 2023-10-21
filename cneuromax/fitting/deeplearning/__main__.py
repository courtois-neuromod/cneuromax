"""Entry point for fitting with Deep Learning."""

import logging
import os
import sys
from importlib import import_module
from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from cneuromax.fitting.common import store_configs as store_base_fitter_configs
from cneuromax.fitting.deeplearning import (
    store_configs as store_deep_learning_configs,
)
from cneuromax.fitting.deeplearning.fitter import (
    DeepLearningFitter,
    DeepLearningFitterHydraConfig,
)


def store_task_configs(cs: ConfigStore) -> None:
    """Store pre-defined task Hydra configurations.

    Parse the task config path from the script arguments, import
    its ``store_configs`` function if it exists, and call it.

    Args:
        cs: .

    Raises:
        ModuleNotFoundError: If the task module cannot be found.
        AttributeError: If the task module does not have a
            ``store_configs`` function.
    """
    for arg in sys.argv:
        if "task" in arg:
            try:
                task_module = import_module(
                    "cneuromax.task." + arg.split("=")[1].split("/")[0],
                )
            except ModuleNotFoundError:
                logging.exception(
                    "The task module cannot be found. Make sure it exists in "
                    "``cneuromax/task`` and is spelled correctly.",
                )

            try:
                task_module.store_configs(cs)
            except AttributeError:
                logging.exception(
                    "The task module must have a ``store_configs`` function. "
                    "Check-out ``cneuromax/tasks/classify_mnist/__init__.py``"
                    "for an example.",
                )

            return

    module_not_found_error_2 = (
        "The task module must be specified in the script "
        "arguments. Example: ``python -m "
        "cneuromax.fitting.deeplearning task=classify_mnist/mlp``."
    )
    raise ModuleNotFoundError(module_not_found_error_2)


def store_configs() -> None:
    """Store configs for the Deep Learning module."""
    cs = ConfigStore.instance()
    store_base_fitter_configs(cs)
    store_deep_learning_configs(cs)
    store_task_configs(cs)


def verify_config(config: DictConfig) -> None:
    """Verifies that various config elements are set correctly.

    Currently, it only makes sure that the ``device`` is set correctly.

    Args:
        config: .
    """
    # Verify device
    if not torch.cuda.is_available():
        logging.info("CUDA is not available, setting device to CPU.")
        config.device = "cpu"


def process_config(config: DictConfig) -> DeepLearningFitterHydraConfig:
    """Process the Hydra config.

    Args:
        config: .

    Returns:
        The processed Hydra config.
    """
    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, value=True)
    dl_config = OmegaConf.to_object(config)
    if not isinstance(dl_config, DeepLearningFitterHydraConfig):
        raise TypeError
    return dl_config


@hydra.main(config_name="config", config_path=".", version_base=None)
def run(config: DictConfig) -> None:
    """.

    Args:
        config: .

    Returns:
        The validation loss.
    """
    verify_config(config)
    dl_config = process_config(config)
    fitter = DeepLearningFitter(dl_config)
    fitter.fit()


def login_wandb() -> None:
    """Login to W&B using the key stored in ``WANDB_KEY.txt``."""
    wandb_key_path = Path(
        str(os.environ.get("CNEUROMAX_PATH")) + "/WANDB_KEY.txt",
    )
    if wandb_key_path.exists():
        with wandb_key_path.open("r") as f:
            key = f.read().strip()
        wandb.login(key=key)
    else:
        logging.info(
            "W&B key not found, proceeding without. You can retrieve your key "
            "from ``https://wandb.ai/settings`` and store it in a file named "
            "``WANDB_KEY.txt`` in the root directory of the project. Discard "
            "this message if you meant not to use W&B.",
        )


if __name__ == "__main__":
    store_configs()
    login_wandb()
    run()
