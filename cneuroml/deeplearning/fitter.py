"""."""
from typing import TYPE_CHECKING

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.distributed import ReduceOp

from cneuroml.deeplearning.utils.lightning import (
    InitOptimParamsCheckpointConnector,
    find_good_batch_size,
    find_good_num_workers,
)

if TYPE_CHECKING:
    from lightning.pytorch.loggers.wandb import WandbLogger
    from lightning.pytorch.trainer import Trainer

    from cneuroml.deeplearning.common.datamodule import BaseDataModule
    from cneuroml.deeplearning.common.litmodule import BaseLitModule

TORCH_COMPILE_MINIMUM_CUDA_VERSION = 7


class Fitter:
    """Deep Learning Fitter.

    This class is the main entry point of the Deep Learning module. It
    acts as an interface between Hydra (configuration + launcher +
    sweeper) and Lightning (trainer + logger + modules).

    Note that this class will be instantiated by ``config.num_nodes`` x
    ``config.num_gpus_per_node`` processes.

    Attributes:
        config (``DictConfig``): The Hydra configuration.
        logger (``WandbLogger``): .
        trainer (``Trainer``): .
        litmodule (``BaseLitModule``): .
        datamodule (``BaseDataModule``): .
    """

    def __init__(self: "Fitter", config: DictConfig) -> None:
        """.

        Transforms the Hydra configuration instructions into Lightning
        objects, sets up hardware-dependent parameters and sets the
        checkpoint path to resume training from (if applicable).

        Args:
            config: The Hydra configuration object.
        """
        self.config = config

        self.instantiate_lightning_objects()
        self.set_batch_size_and_num_workers()
        self.set_checkpoint_path()

    def instantiate_lightning_objects(self: "Fitter") -> None:
        """.

        Instantiates:
        - the W&B Logger (in offline mode if running on Slurm)
        - the Lightning Trainer
        - the Lightning Module (compiled if using CUDA 7.0+ GPUs)
        - the Lightning DataModule
        """
        self.logger: WandbLogger = instantiate(
            self.config.logger,
            offline=(
                HydraConfig.get().launcher._target_
                == "hydra_plugins.hydra_submitit_launcher.submitit_launcher."
                "SlurmLauncher"
            ),
        )
        self.trainer: Trainer = instantiate(
            self.config.trainer,
            logger=self.logger,
        )

        self.litmodule: BaseLitModule = instantiate(
            self.config.litmodule,
        )

        if (
            self.config.num_gpus_per_node > 0
            and torch.cuda.get_device_capability()[0]
            >= TORCH_COMPILE_MINIMUM_CUDA_VERSION
        ):
            # mypy: torch.compile isn't typed for LightningModule
            self.litmodule = torch.compile(  # type: ignore [assignment]
                self.litmodule,
            )

        self.datamodule: BaseDataModule = instantiate(
            self.config.datamodule,
        )

    def set_batch_size_and_num_workers(self: "Fitter") -> None:
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
        if not self.config.load_path_hpo:
            this_gpu_good_batch_size = find_good_batch_size(self.config)
            this_gpu_good_num_workers = find_good_num_workers(
                self.config,
                this_gpu_good_batch_size,
            )

        batch_size = int(
            self.trainer.strategy.reduce(
                torch.tensor(this_gpu_good_batch_size),
                reduce_op=ReduceOp.MIN,  # type: ignore [arg-type]
            ),
        )

        num_workers = int(
            self.trainer.strategy.reduce(
                torch.tensor(this_gpu_good_num_workers),
                reduce_op=ReduceOp.MAX,  # type: ignore [arg-type]
            ),
        )

        self.datamodule.config.per_device_batch_size = batch_size
        self.datamodule.config.per_device_num_workers = num_workers

    def set_checkpoint_path(self: "Fitter") -> None:
        """Sets the path to the checkpoint to resume training from.

        Three cases are considered:
        - if the ``config.load_path_hpo`` parameter is set, we are
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
        if self.config.load_path_hpo:
            self.ckpt_path = self.config.load_path_hpo
            self.trainer._checkpoint_connector = (
                InitOptimParamsCheckpointConnector(self.trainer)
            )

        elif self.config.load_path:
            self.ckpt_path = self.config.load_path

        else:
            self.ckpt_path = None

    def fit(self: "Fitter") -> float:
        """Fitting method.

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

        self.trainer.save_checkpoint(self.config.save_path)

        return self.trainer.validate(
            model=self.litmodule,
            datamodule=self.datamodule,
        )[0]["val/loss"]
