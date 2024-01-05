""":mod:`hydra-core` utilities."""
from collections.abc import Callable
from typing import Any

from hydra.core.hydra_config import HydraConfig
from hydra_plugins.hydra_submitit_launcher.config import (
    LocalQueueConf,
    SlurmQueueConf,
)
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    LocalLauncher,
)
from omegaconf import DictConfig, OmegaConf


def get_path(clb: Callable[..., Any]) -> str:
    """Returns the path to the input callable.

    Args:
        clb: The callable to retrieve the path for.

    Returns:
        The full module path to :paramref:`clb`.
    """
    return f"{clb.__module__}.{clb.__name__}"


def get_launcher_config() -> LocalQueueConf | SlurmQueueConf:
    """Retrieves/validates this job's :mod:`hydra-core` launcher config.

    Returns:
        The :mod:`hydra-core` launcher config.
    """
    launcher_dict_config: DictConfig = HydraConfig.get().launcher
    launcher_container_config = OmegaConf.to_container(
        launcher_dict_config,
    )
    if not isinstance(launcher_container_config, dict):
        raise TypeError
    launcher_config_dict = dict(launcher_container_config)
    return (
        LocalQueueConf(**launcher_config_dict)
        if launcher_dict_config._target_  # noqa: SLF001
        == get_path(LocalLauncher)
        else SlurmQueueConf(**launcher_config_dict)
    )
