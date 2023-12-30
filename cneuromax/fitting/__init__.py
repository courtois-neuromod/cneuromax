"""Fitting module."""

from typing import Any

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf

from cneuromax.fitting.config import BaseFittingHydraConfig

__all__ = [
    "store_base_fitting_configs",
    "store_launcher_configs",
]


def store_base_fitting_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` root fitting configs.

    Args:
        cs: See :paramref:`~cneuromax.store_task_configs.cs`.
    """
    store_launcher_configs(cs)
    cs.store(name="base_fitting", node=BaseFittingHydraConfig)


def store_launcher_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` ``hydra/launcher`` group configs.

    Config names: ``submitit_slurm_acan``,\
        ``submitit_slurm_acan_simexp``.

    Args:
        cs: See :paramref:`~cneuromax.store_task_configs.cs`.
    """
    cs.store(name="setup_apptainer_acan", node=["module load apptainer"])
    setup: Any = "${merge:${setup_apptainer_acan},${copy_data_commands}}"
    python = "apptainer --nv exec ${oc.env:SCRATCH}/cneuromax.sif python"
    cs.store(
        group="hydra/launcher",
        name="submitit_slurm_acan",
        node=SlurmQueueConf(python=python, setup=setup),
    )
    cs.store(
        group="hydra/launcher",
        name="submitit_slurm_acan_simexp",
        node=SlurmQueueConf(account="rrg-pbellec", setup=setup),
    )
