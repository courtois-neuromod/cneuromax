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
        run_dir: Path to the run's directory. Every artifact generated\
            during the run will be stored in this directory.
        data_dir: Path to the data directory. This directory is\
            typically shared between runs. It is used to store\
            datasets, pre-trained models, etc.
        run_dir_exist_ok: Whether the run directory is allowed to\
            already exist. If ``False`` and the run directory already\
            exists, it will be renamed to ``{run_dir}_i``,\
            where ``i`` is the smallest non-zero integer such that\
            ``{run_dir}_i`` does not exist. Generally, do not modify\
            this parameter unless you want to resume a run.
    """

    run_dir: An[str, not_empty()] = "data/untitled_run/"
    data_dir: An[str, not_empty()] = "${run_dir}/../"
    run_dir_exist_ok: bool = False


def pre_process_base_config(config: DictConfig) -> None:
    """Validates raw task config before it is made structured.

    Makes sure that the ``run_dir`` does not already exist if
    ``run_dir_exist_ok`` is ``False``. It loops through ``{run_dir}_1``,
    ``{run_dir}_2``, etc until it finds a directory that does not exist.

    Args:
        config: The raw task config.
    """
    if config.run_dir_exist_ok:
        return
    run_dir = config.run_dir
    path = Path(run_dir)
    if path.exists():
        i = 1
        while path.exists():
            path = Path(run_dir + f"_{i}")
            i += 1
        logging.info(
            f"{Path(run_dir).absolute()} already exists, "
            f"changing it to {path.absolute()}.",
        )
        run_dir = str(path)


T = TypeVar("T", bound=BaseHydraConfig)


def process_config(config: DictConfig, structured_config_class: type[T]) -> T:
    """Turns the raw task config into a structured config.

    Args:
        config: See :paramref:`pre_process_base_config.config`.
        structured_config_class: The structured config class to turn\
            the raw config into.

    Returns:
        The processed structured Hydra config.
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
        config: The processed :mod:`hydra-core` config.
    """
    path = Path(config.run_dir)
    if not path.exists():
        logging.info(
            f"Creating the run directory {path.absolute()} as it does not "
            "exist.",
        )
        path.mkdir(parents=True)
