""":class:`FittingTaskRunner`."""
from typing import Any

from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from hydra_zen import ZenStore

# from cneuromax.fitting.config import FittingTaskConfig
from cneuromax.runner import BaseTaskRunner


def store_launcher_configs(store: ZenStore) -> None:
    """Stores Hydra ``hydra/launcher`` group configs.

    Names: ``submitit_slurm_acan``, ``submitit_slurm_acan_simexp``.

    Args:
        store: See :meth:`~.BaseTaskRunner.store_configs`.
    """
    store(["module load apptainer"], name="setup_apptainer_acan")
    setup: Any = "${merge:${setup_apptainer_acan},${copy_data_commands}}"
    python = "apptainer --nv exec ${oc.env:SCRATCH}/cneuromax.sif python"
    store(
        SlurmQueueConf(python=python, setup=setup),
        group="hydra/launcher",
        name="submitit_slurm_acan",
    )
    store(
        SlurmQueueConf(account="rrg-pbellec", setup=setup),
        group="hydra/launcher",
        name="submitit_slurm_acan_simexp",
    )


class FittingTaskRunner(BaseTaskRunner):
    """Fitting ``task`` runner.

    Attr:
        subtask_config: See :attr:`~.BaseTaskRunner.subtask_config`.
    """

    @classmethod
    def store_configs(cls: type["FittingTaskRunner"], store: ZenStore) -> None:
        """Stores structured configs.

        .. warning::

            Make sure to call this method if you are overriding it.

        Args:
            store: See :meth:`~.BaseTaskRunner.store_configs`.
        """
        super().store_configs(store)
        store_launcher_configs(store)
