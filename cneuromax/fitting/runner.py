""":class:`BaseTaskRunner`."""
from typing import Any

import torch
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from hydra_zen import ZenStore

from cneuromax.fitting.config import FittingSubtaskConfig
from cneuromax.runner import BaseTaskRunner


def store_base_fitting_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` root fitting configs.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store_launcher_configs(cs)
    cs.store(name="base_fitting", node=FittingConfig)


def store_launcher_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``hydra/launcher`` group configs.

    Config names: ``submitit_slurm_acan``,\
        ``submitit_slurm_acan_simexp``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """


class FittingTaskRunner(BaseTaskRunner):
    """Fitting ``task`` runner.

    Attr:
        subtask_config: See :attr:`~.BaseTaskRunner.subtask_config`.
    """

    subtask_config: type[FittingSubtaskConfig] = FittingSubtaskConfig

    @staticmethod
    def store_configs(store: ZenStore) -> None:
        """Stores structured configs.

        .. warning::

            Make sure to call this method if you are overriding it.

        Args:
            store: See :meth:`~.BaseTaskRunner.store_configs`.
        """
        BaseTaskRunner.store_configs(store)
        # Launcher configs
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
        # Fitting config
        store(FittingSubtaskConfig, name="fitting")

    @staticmethod
    def validate_subtask_config(
        config: FittingSubtaskConfig,  # type: ignore[override]
    ) -> None:
        """See :meth:`BaseTaskRunner.validate_subtask_config`.

        Args:
            config: See :attr:`subtask_config`

        Raises:
            RuntimeError: If ``FittingSubtaskConfig.device`` is set to
                ``gpu`` but CUDA is not available.
        """
        BaseTaskRunner.validate_subtask_config(config)
        if not torch.cuda.is_available() and config.device == "gpu":
            error_msg = "CUDA is not available, but device is set to GPU."
            raise RuntimeError(error_msg)
