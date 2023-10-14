"""Fitting common Hydra configuration creation & storage."""

from typing import Any

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf

from cneuromax.fitting.common.fitter import BaseFitterHydraConfig

__all__ = ["BaseFitterHydraConfig", "store_configs"]


def store_configs(cs: ConfigStore) -> None:
    """Stores all common Hydra fitting related configs.

    Args:
        cs: .
    """
    store_launcher_configs(cs)
    cs.store(name="base_fitter", node=BaseFitterHydraConfig)


def store_launcher_configs(cs: ConfigStore) -> None:
    """Stores Hydra ``hydra/launcher`` group configs.

    Names: ``submitit_slurm_acan``, ``submitit_slurm_acan_simexp``.

    Args:
        cs: .
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
