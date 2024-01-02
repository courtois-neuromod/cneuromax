"""Root :mod:`hydra-core` config & utilities."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated as An
from typing import TypeVar

from omegaconf import DictConfig, OmegaConf

from cneuromax.utils.annotations import not_empty


@dataclass(frozen=True)
class BaseHydraConfig:
    """Base structured :mod:`hydra-core` configuration.

    Args:
        task_run_dir: Path to the task run's directory. Every artifact\
            generated while running the task will be stored in this\
            directory.
        data_dir: Path to the data directory. This directory is\
            typically shared between task runs. It is used to store\
            datasets, pre-trained models, etc.
    """

    task_run_dir: An[str, not_empty()] = "${hydra:runtime.output_dir}"
    data_dir: An[str, not_empty()] = "${oc.env:CNEUROMAX_PATH}/data/"


def pre_process_base_config(config: DictConfig) -> None:
    """Validates raw task config before it is made structured.

    Args:
        config: The "raw" config returned by the :mod:`hydra-core`\
            :func:`main` decorator.
    """
    path = Path(config.task_run_dir)
    if path.exists():
        logging.info(
            f"The task run directory {path.absolute()} exists. Ignore this "
            "message if it appears in the middle of a task run. However, if "
            "you are starting a new task run, the previous task run's "
            "artifacts will possibly be overwritten.",
        )
    else:
        path.mkdir(parents=True)


T = TypeVar("T", bound=BaseHydraConfig)


def process_config(config: DictConfig, structured_config_class: type[T]) -> T:
    """Turns the raw task config into a structured config.

    Args:
        config: See :paramref:`~.pre_process_base_config.config`.
        structured_config_class: The structured config class to turn\
            the raw config into.

    Returns:
        See :paramref:`~.post_process_base_config.config`.
    """
    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, value=True)
    config = OmegaConf.to_object(config)
    if not isinstance(config, structured_config_class):
        error_msg = (
            f"The config must be an instance of {structured_config_class}"
            f", not{type(config)}."
        )
        raise TypeError(error_msg)
    return config


def post_process_base_config(config: BaseHydraConfig) -> None:
    """Validates the structured task config.

    Creates the run directory if it does not exist.

    Args:
        config: The structured :mod:`hydra-core` config used throughout\
            the execution that was resolved in :func:`process_config`.
    """
    path = Path(config.task_run_dir)
    if not path.exists():
        logging.info(
            f"Creating the run directory {path.absolute()} as it does not "
            "exist.",
        )
        path.mkdir(parents=True)
