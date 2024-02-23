r"""Fitting :mod:`hydra` config storing."""

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
    # Setting up the launchers is a little bit different from the other
    # configs. Fields get resolved before the ``subtask`` is created.
    args: dict[str, Any] = {  # `fs_builds`` does not like dict[str, str]
        "submitit_folder": "${hydra.sweep.dir}/${now:%Y-%m-%d-%H-%M-%S}/",
        "stderr_to_stdout": True,
        "timeout_min": 1440,  # 24 hours
    }
    store(LocalQueueConf(**args), group="hydra/launcher", name="local")
    args.update(
        {
            "setup": "${merge:${setup_apptainer_slurm},${copy_data_commands}}",
            "python": ""
            "apptainer --nv exec ${oc.env:SCRATCH}/cneuromax.sif python",
        },
    )
    store(SlurmQueueConf(**args), group="hydra/launcher", name="slurm")
    store(["module load apptainer"], name="setup_apptainer_slurm")
