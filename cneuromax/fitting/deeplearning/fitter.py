"""Deep Learning Fitter Config & Class."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    SlurmLauncher,
)
from omegaconf import MISSING
from torch.distributed import ReduceOp

from cneuromax.fitting.common import BaseFitterHydraConfig
from cneuromax.utils.hydra import get_path

if TYPE_CHECKING:
    from hydra_plugins.hydra_submitit_launcher.config import (
        LocalQueueConf,
        SlurmQueueConf,
    )
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import Logger

    from cneuromax.fitting.deeplearning.datamodule import BaseDataModule
    from cneuromax.fitting.deeplearning.litmodule import (
        BaseLitModule,
    )

TORCH_COMPILE_MINIMUM_CUDA_VERSION = 7


@dataclass
class DeepLearningFitterHydraConfig(BaseFitterHydraConfig):
    """.

    Attributes:
        trainer: .
        litmodule: .
        datamodule: .
        logger: .
    """

    trainer: Any = MISSING
    litmodule: Any = MISSING
    datamodule: Any = MISSING
    logger: Any = MISSING


class DeepLearningFitter:
    """Deep Learning Fitter.

    This class is the main entry point of the Deep Learning module. It
    acts as an interface between Hydra (configuration + launcher +
    sweeper) and Lightning (trainer + logger + modules).

    Note that this class will be instantiated by ``config.num_nodes`` x
    ``config.gpus_per_node`` processes.

    Attributes:
        config (``DeepLearningFitterHydraConfig``): .
        logger (``Logger``): .
        trainer (``Trainer``): .
        litmodule (``BaseLitModule``): .
        datamodule (``BaseDataModule``): .
    """

    def __init__(
        self: "DeepLearningFitter",
        config: DeepLearningFitterHydraConfig,
    ) -> None:
        """Constructor, stores config and initializes various objects.

        Transforms the Hydra configuration instructions into Lightning
        objects, sets up hardware-dependent parameters and sets the
        checkpoint path to resume training from (if applicable).

        Args:
            config: .
        """
        self.config = config
        self.launcher_config: LocalQueueConf | SlurmQueueConf = (
            HydraConfig.get().launcher
        )
        self.instantiate_lightning_objects()
        self.set_batch_size_and_num_workers()
        self.set_checkpoint_path()

    def instantiate_lightning_objects(self: "DeepLearningFitter") -> None:
        """."""
        """
        kwargs = {}
        if self.config.logger._target_ == get_path(WandbLogger):
            kwargs["offline"] = self.launcher_config._target_ = get_path
                SlurmLauncher,
            )
        """
        self.logger: Logger | None = instantiate(
            self.config.logger,
        )  # **kwargs
        self.logger = (
            None
            if self.launcher_config._target_ == get_path(SlurmLauncher)
            else self.logger
        )

        callbacks = None

        """
        if self.launcher_config._target_ == get_path(SlurmLauncher):
            callbacks = [TriggerWandbSyncLightningCallback()]
        """

        self.trainer: Trainer = instantiate(
            config=self.config.trainer,
            devices=self.launcher_config.gpus_per_node or 1
            if self.config.trainer.accelerator == "gpu"
            else self.launcher_config.tasks_per_node,
            logger=self.logger,
            callbacks=callbacks,
        )

        self.datamodule: BaseDataModule = instantiate(self.config.datamodule)

        self.litmodule: BaseLitModule = instantiate(self.config.litmodule)
        if (
            torch.cuda.get_device_capability()[0]
            >= TORCH_COMPILE_MINIMUM_CUDA_VERSION
        ):
            self.litmodule = torch.compile(  # type: ignore [assignment]
                self.litmodule,  # mypy: torch.compile not typed for Lightning.
            )

    def set_batch_size_and_num_workers(self: "DeepLearningFitter") -> None:
        """.

        If starting a new HPO run, finds and sets "good" ``batch_size``
        and ``num_workers`` parameters.

        See the ``find_good_batch_size`` and ``find_good_num_workers``
        functions documentation for more details.

        We make the assumption that if we are resuming from a checkpoint
        created while running hyper-parameter optimization, we are
        running on the same hardware configuration as was used to create
        the checkpoint. Therefore, we do not need to once again look for
        good ``batch_size`` and ``num_workers`` parameters.
        """
        from cneuromax.fitting.deeplearning.utils.lightning import (
            find_good_num_workers,
            find_good_per_device_batch_size,
        )

        if not self.config.load_path_pbt:
            proposed_per_device_batch_size: int = (
                find_good_per_device_batch_size(
                    self.config,
                    self.launcher_config,
                )
            )
            proposed_per_device_num_workers: int = find_good_num_workers(
                self.config,
                self.launcher_config,
                proposed_per_device_batch_size,
            )

        per_device_batch_size: int = int(
            self.trainer.strategy.reduce(
                torch.tensor(proposed_per_device_batch_size),
                reduce_op=ReduceOp.MIN,  # type: ignore [arg-type]
            ),
        )

        per_device_num_workers: int = int(
            self.trainer.strategy.reduce(
                torch.tensor(proposed_per_device_num_workers),
                reduce_op=ReduceOp.MAX,  # type: ignore [arg-type]
            ),
        )

        self.datamodule.per_device_batch_size = per_device_batch_size
        self.datamodule.per_device_num_workers = per_device_num_workers

    def set_checkpoint_path(self: "DeepLearningFitter") -> None:
        """Sets the path to the checkpoint to resume training from.

        Three cases are considered:
        - if the ``config.load_path_pbt`` parameter is set, we are
        resuming from a checkpoint created while running HPO. In this
        case, we set the checkpoint path to the value of
        ``config.load_path_hpo`` and use a custom checkpoint connector
        to not override the new HPO config values.
        - if the ``config.load_path`` parameter is set (but not
        ``config.load_path_hpo``), we are resuming from a regular
        checkpoint. In this case, we set the checkpoint path to the
        value of ``config.load_path``.
        - if neither ``config.load_path_hpo`` nor ``config.load_path``
        are set, we are starting a new training run. In this case, we
        set the checkpoint path to ``None``.
        """
        self.ckpt_path: str | None

        if self.config.load_path_pbt:
            from cneuromax.fitting.deeplearning.utils.lightning import (
                InitOptimParamsCheckpointConnector,  # Prevent circular import
            )

            self.ckpt_path = self.config.load_path_pbt
            self.trainer._checkpoint_connector = (
                InitOptimParamsCheckpointConnector(self.trainer)
            )
        elif self.config.load_path:
            self.ckpt_path = self.config.load_path
        else:
            self.ckpt_path = None

    def fit(self: "DeepLearningFitter") -> float:
        """.

        Trains (or resumes training) the model, saves a checkpoint and
        returns the final validation loss.

        Returns:
            The final validation loss.
        """
        self.trainer.fit(
            model=self.litmodule,
            datamodule=self.datamodule,
            ckpt_path=self.ckpt_path,
        )
        if self.config.save_path:
            self.trainer.save_checkpoint(self.config.save_path)
        return self.trainer.validate(
            model=self.litmodule,
            datamodule=self.datamodule,
        )[0]["val/loss"]
