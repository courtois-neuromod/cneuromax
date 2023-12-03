"""Root :mode:`cneuromax` :mod:`hydra-core` configuration management."""

import logging
import sys
from importlib import import_module
from typing import Annotated as An
from typing import TypeVar

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from cneuromax.utils.annotations import not_empty


class BaseHydraConfig:
    """Base structured :mod:`hydra-core` configuration.

    Attributes:
        data_dir: Path to the root data directory.
    """

    data_dir: An[str, not_empty()] = "data/untitled_run/"


T = TypeVar("T", bound=BaseHydraConfig)


def process_config(dict_config: DictConfig, config_class: type[T]) -> T:
    """Process the Hydra config.

    Args:
        dict_config: The Hydra config.
        config_class: The class of the Hydra config.

    Returns:
        The processed Hydra config.
    """
    OmegaConf.resolve(dict_config)
    OmegaConf.set_struct(dict_config, value=True)
    config = OmegaConf.to_object(dict_config)
    if not isinstance(config, config_class):
        raise TypeError
    return config


def store_task_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` task configurations.

    Parses the task config path from the script arguments, import
    its ``store_configs`` function if it exists, and calls it.

    Args:
        cs: A singleton instance that manages the :mod:`hydra-core`\
            configuration store.

    Raises:
        ModuleNotFoundError: If the task module cannot be found.
        AttributeError: If the task module does not have a\
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
                    "`cneuromax/task` and is spelled correctly. If it does "
                    "exist, make sure it an `__init__.py` file exists in "
                    "its directory.",
                )
                raise
            try:
                task_module.store_configs(cs)
            except AttributeError:
                logging.exception(
                    "The task module `store_configs` function cannot be "
                    "found. Make sure it exists in your `__init__.py` file. "
                    "Check-out `cneuromax/tasks/classify_mnist/__init__.py`"
                    "for an example.",
                )
                raise

            return

    module_not_found_error_2 = (
        "The task must be specified in the script arguments. Example: "
        "`python -m cneuromax.fitting.deeplearning task=classify_mnist/mlp`."
    )
    raise ModuleNotFoundError(module_not_found_error_2)
