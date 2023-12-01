""":mod:`cneuromax.fitting` :mod:`hydra-core` config & validation."""

import logging
from dataclasses import dataclass
from typing import Annotated as An

import torch

from cneuromax import BaseHydraConfig
from cneuromax.utils.annotations import not_empty, one_of


@dataclass
class BaseFittingHydraConfig(BaseHydraConfig):
    """Base structured :mod:`hydra-core` fitting config.

    Attributes:
        device: Computing device to use for large matrix operations.
        model_load_path: Path to the checkpointed model to load.
        pbt_load_path: Path to the HPO checkpoint to load for PBT.
        model_save_path: Path to save the model to.
        copy_data_commands: List of commands to execute to copy data\
            for the purpose of fitting.
    """

    device: An[str, one_of("cpu", "gpu")] = "cpu"
    model_load_path: An[str, not_empty()] | None = None
    pbt_load_path: An[str, not_empty()] | None = None
    model_save_path: An[str, not_empty()] = "${data_dir}/lightning/final.ckpt"
    copy_data_commands: list[str] | None = None


def verify_config(config: BaseFittingHydraConfig) -> None:
    """Verifies that various config elements are set correctly.

    Args:
        config: The run's config, see\
            :class:`BaseFittingHydraConfig`.

    """
    if not torch.cuda.is_available():
        logging.info("CUDA is not available, setting device to CPU.")
        config.device = "cpu"
