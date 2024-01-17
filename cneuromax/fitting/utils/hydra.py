""":mod:`hydra-core` utilities."""
from hydra._internal.core_plugins.basic_launcher import (
    BasicLauncher,
)
from hydra.core.hydra_config import HydraConfig
from hydra_plugins.hydra_submitit_launcher.config import (
    LocalQueueConf,
    SlurmQueueConf,
)
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    LocalLauncher,
    SlurmLauncher,
)
from omegaconf import DictConfig, OmegaConf

from cneuromax.utils.misc import get_path


def get_launcher_config() -> LocalQueueConf | SlurmQueueConf:
    """Retrieves/validates this job's :mod:`hydra-core` launcher config.

    Returns:
        The :mod:`hydra-core` launcher config.

    Raises:
        TypeError: If the launcher config is not a dict or if the\
            launcher is not supported.
    """
    launcher_dict_config: DictConfig = HydraConfig.get().launcher
    launcher_container_config = OmegaConf.to_container(
        cfg=launcher_dict_config,
    )
    if not isinstance(launcher_container_config, dict):
        raise TypeError
    launcher_config_dict = dict(launcher_container_config)
    if launcher_dict_config._target_ == get_path(  # noqa: SLF001
        LocalLauncher,
    ):
        return LocalQueueConf(**launcher_config_dict)
    if launcher_dict_config._target_ == get_path(  # noqa: SLF001
        SlurmLauncher,
    ):
        return SlurmQueueConf(**launcher_config_dict)
    if launcher_dict_config._target_ == get_path(  # noqa: SLF001
        BasicLauncher,
    ):
        error_msg = (
            "`hydra/launcher: basic` (the default launcher) is not supported. "
            "Use `override hydra/launcher: submitit_local` or "
            "`override hydra/launcher: submitit_slurm`."
        )
        raise TypeError(error_msg)
    error_msg = (
        "Unsupported launcher: "
        f"{launcher_dict_config._target_}"  # noqa: SLF001
    )
    raise TypeError(error_msg)
