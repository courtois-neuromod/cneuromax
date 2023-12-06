"""Root :mod:`hydra-core` config, its task extensions & validation."""

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
    """

    run_dir: An[str, not_empty()] = "data/untitled_run/"
    data_dir: An[str, not_empty()] = "${run_dir}/../"


def pre_process_base_config(dict_config: DictConfig) -> None:
    """Pre-processes config from :func:`hydra.main` before resolution.

    Makes sure that the ``run_dir`` does not already exist. If it does,
    it loops through ``{run_dir}_1``, ``{run_dir}_2``, etc. until it
    finds a directory that does not exist.

    Args:
        dict_config: The raw config retrieved through the\
            :func:`hydra.main` decorator.
    """
    run_dir = dict_config.run_dir
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


def process_config(dict_config: DictConfig, structured_config: type[T]) -> T:
    """Turns config from :func:`hydra.main` into a structured config.

    Args:
        dict_config: See\
            :paramref:`pre_process_base_config.dict_config`.
        structured_config: The dataclass to instantiate from the\

    Returns:
        The processed Hydra config.
    """
    OmegaConf.resolve(dict_config)
    OmegaConf.set_struct(dict_config, value=True)
    config = OmegaConf.to_object(dict_config)
    if not isinstance(config, structured_config):
        error_msg = (
            f"The config must be an instance of {structured_config}, not "
            f"{type(config)}."
        )
        raise TypeError(error_msg)
    return config


def post_process_base_config(config: BaseHydraConfig) -> None:
    """Post-processes the :mod:`hydra-core` config after it is resolved.

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
