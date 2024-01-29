""":func:`train`."""

from functools import partial

from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from cneuromax.fitting.config import (
    FittingSubtaskConfig,
)
from cneuromax.fitting.deeplearning.datamodule import BaseDataModule
from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.fitting.deeplearning.utils.lightning import (
    instantiate_trainer_and_logger,
    set_batch_size_and_num_workers,
    set_checkpoint_path,
)


def train(
    trainer: partial[Trainer],
    datamodule: BaseDataModule,
    litmodule: BaseLitModule,
    logger: partial[WandbLogger],
    config: FittingSubtaskConfig,
) -> float:
    """Trains a Deep Neural Network.

    Note that this function will be executed by
    ``num_nodes * gpus_per_node`` processes/tasks. Those variables are
    set in the Hydra launcher configuration.

    Trains (or resumes training) the model, saves a checkpoint and
    returns the final validation loss.

    Args:
        trainer: See :class:`~lightning.pytorch.Trainer`.
        datamodule: See :class:`.BaseDataModule`.
        litmodule: See :class:`.BaseLitModule`.
        logger: See\
            :class:`~lightning.pytorch.loggers.wandb.WandbLogger`.
        config: See :paramref:`~.FittingSubtaskConfig`.

    Returns:
        The final validation loss.
    """
    full_trainer, full_logger = instantiate_trainer_and_logger(
        partial_trainer=trainer,
        partial_logger=logger,
        device=config.device,
    )
    """TODO: Add logic for HPO"""
    set_batch_size_and_num_workers(
        trainer=full_trainer,
        datamodule=datamodule,
        litmodule=litmodule,
        device=config.device,
        output_dir=config.output_dir,
    )
    ckpt_path = set_checkpoint_path(trainer=full_trainer, config=config)
    full_trainer.fit(
        model=litmodule,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )
    """TODO: Add logic for HPO
    trainer.save_checkpoint(filepath=config.model_load_path)
    """
    return full_trainer.validate(model=litmodule, datamodule=datamodule)[0][
        "val/loss"
    ]
