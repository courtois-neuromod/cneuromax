""":mod:`hydra-core` utilities."""
from dataclasses import dataclass
from typing import Annotated as An

from hydra._internal.core_plugins.basic_launcher import (
    BasicLauncher,
    BasicLauncherConf,
)
from hydra.core.hydra_config import HydraConfig
from hydra_plugins.hydra_submitit_launcher.config import (
    LocalQueueConf,
    SlurmQueueConf,
)
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    LocalLauncher,
)
from omegaconf import DictConfig, OmegaConf

from cneuromax.utils.beartype import equal
from cneuromax.utils.misc import get_path


@dataclass
class BaseLauncherConf(BasicLauncherConf):
    """:class:`.BasicLauncherConf` w/ CPU & GPU info.

    These fields do not impact the launcher itself, but are instead
    meant to hold that information for other objects that might depend
    on it. Follows the same naming conventions as
    :class:`.SlurmQueueConf` & :class:`.LocalQueueConf`.

    Args:
        nodes: Number of nodes to use for the job. Must be 1.
        tasks_per_node: Number of tasks/processes to spawn for the job.
        cpus_per_task: Number of CPUs to use per task/process.
        gpus_per_node: Number of GPUs to use for the job.
    """

    nodes: An[int, equal(1)] = 1
    tasks_per_node: int = 1
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None


def get_launcher_config() -> (
    BaseLauncherConf | LocalQueueConf | SlurmQueueConf
):
    """Retrieves/validates this job's :mod:`hydra-core` launcher config.

    Returns:
        The :mod:`hydra-core` launcher config.
    """
    launcher_dict_config: DictConfig = HydraConfig.get().launcher
    launcher_container_config = OmegaConf.to_container(
        cfg=launcher_dict_config,
    )
    if not isinstance(launcher_container_config, dict):
        raise TypeError
    launcher_config_dict = dict(launcher_container_config)
    if launcher_dict_config._target_ == get_path(  # noqa: SLF001
        BasicLauncher,
    ):
        return BaseLauncherConf(**launcher_config_dict)
    if launcher_dict_config._target_ == get_path(  # noqa: SLF001
        LocalLauncher,
    ):
        return LocalQueueConf(**launcher_config_dict)
    # launcher_dict_config._target_ == get_path(SlurmLauncher):
    return SlurmQueueConf(**launcher_config_dict)
