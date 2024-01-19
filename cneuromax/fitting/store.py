r"""Fitting :mod:`hydra-core` config storing."""
from typing import Any

from hydra_plugins.hydra_submitit_launcher.config import (
    LocalQueueConf,
    SlurmQueueConf,
)
from hydra_zen import ZenStore


def store_launcher_configs(store: ZenStore) -> None:
    """Stores Hydra ``hydra/launcher`` group configs.

    Names: ``local``, ``slurm``.

    Args:
        store: See :meth:`~.BaseTaskRunner.store_configs`.
    """
    submitit_folder = "${hydra.sweep.dir}/${now:%H-%M-%S}"
    store(
        LocalQueueConf(
            submitit_folder=submitit_folder,
        ),
        group="hydra/launcher",
        name="local",
    )
    store(["module load apptainer"], name="setup_apptainer_slurm")
    setup: Any = "${merge:${setup_apptainer_slurm},${copy_data_commands}}"
    python = "apptainer --nv exec ${oc.env:SCRATCH}/cneuromax.sif python"
    store(
        SlurmQueueConf(
            submitit_folder=submitit_folder,
            python=python,
            setup=setup,
        ),
        group="hydra/launcher",
        name="slurm",
    )
