"""Root :mod:`hydra-core` fitting config & validation."""

import logging
from dataclasses import dataclass
from typing import Annotated as An

import torch
from omegaconf import DictConfig

from cneuromax.config import (
    BaseHydraConfig,
    post_process_base_config,
    pre_process_base_config,
)
from cneuromax.utils.annotations import one_of


@dataclass
class BaseFittingHydraConfig(BaseHydraConfig):
    """Base structured :mod:`hydra-core` fitting config.

    Args:
        device: Computing device to use for large matrix operations.
        copy_data_commands: List of commands to execute to transfer the\
            training data to the\
            :paramref:`cneuromax.config.BaseHydraConfig.data_dir`\
            directory. This is useful when the training data is stored\
            on a disk that is different from the one used by the\
            training machine.
    """

    device: An[str, one_of("cpu", "gpu")] = "cpu"
    copy_data_commands: list[str] | None = None


def pre_process_base_fitting_config(config: DictConfig) -> None:
    """Pre-processes config from :func:`hydra.main` before resolution.

    Used for changing the computing device if CUDA is not available.

    Args:
        config: The not yet processed :mod:`hydra-core` config.

    """
    pre_process_base_config(config)
    if not torch.cuda.is_available():
        logging.info("CUDA is not available, setting device to CPU.")
        config.device = "cpu"


def post_process_base_fitting_config(
    config: BaseFittingHydraConfig,
) -> None:
    """Post-processes the :mod:`hydra-core` config after it is resolved.

    Args:
        config: The processed :mod:`hydra-core` config.
    """
    post_process_base_config(config)
