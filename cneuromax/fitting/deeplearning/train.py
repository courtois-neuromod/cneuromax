""":func:`.train`."""

from functools import partial

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from cneuromax.fitting.deeplearning.config import (
    DeepLearningSubtaskConfig,
)
from cneuromax.fitting.deeplearning.datamodule import BaseDataModule
from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.fitting.deeplearning.utils.lightning import (
    instantiate_trainer,
    set_batch_size_and_num_workers,
)
from cneuromax.utils.misc import seed_all

TORCH_COMPILE_MINIMUM_CUDA_VERSION = 7


def train(
    trainer: partial[Trainer],
    datamodule: BaseDataModule,
    litmodule: BaseLitModule,
    logger: partial[WandbLogger],
    config: DeepLearningSubtaskConfig,
) -> float:
    """Trains a Deep Neural Network.

    Note that this function will be executed by
    ``num_nodes * gpus_per_node`` processes/tasks. Those variables are
    set in the Hydra launcher configuration.

    Trains (or resumes training) the model, saves a checkpoint and
    returns the final validation loss.

    Args:
        trainer
        datamodule
        litmodule
        logger
        config

    Returns:
        The final validation loss.
    """
    seed_all(config.seed)
    trainer: Trainer = instantiate_trainer(
        trainer_partial=trainer,
        logger_partial=logger,
        device=config.device,
        output_dir=config.output_dir,
        save_every_n_train_steps=config.save_every_n_train_steps,
    )
    """TODO: Add logic for HPO"""
    set_batch_size_and_num_workers(
        trainer=trainer,
        datamodule=datamodule,
        litmodule=litmodule,
        device=config.device,
        output_dir=config.output_dir,
    )
    if (
        config.compile
        and config.device == "gpu"
        and torch.cuda.get_device_capability()[0]
        >= TORCH_COMPILE_MINIMUM_CUDA_VERSION
    ):
        litmodule.nnmodule = torch.compile(litmodule.nnmodule)  # type: ignore [assignment]
    litmodule.trainer = trainer
    trainer.fit(
        model=litmodule,
        datamodule=datamodule,
        ckpt_path=config.ckpt_path,
    )
    """TODO: Add logic for HPO"""
    return trainer.validate(model=litmodule, datamodule=datamodule)[0][
        "val/loss"
    ]
