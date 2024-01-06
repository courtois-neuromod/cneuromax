""":func:`train`."""
from functools import partial

from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from cneuromax.fitting.deeplearning.config import (
    DeepLearningSubtaskConfig,
)
from cneuromax.fitting.deeplearning.datamodule import BaseDataModule
from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.fitting.deeplearning.utils.lightning import (
    instantiate_lightning_objects,
    set_batch_size_and_num_workers,
    set_checkpoint_path,
)
from cneuromax.utils.hydra import get_launcher_config


def train(
    trainer: partial[Trainer],
    logger: partial[WandbLogger],
    datamodule: BaseDataModule,
    litmodule: BaseLitModule,
    config: DeepLearningSubtaskConfig,
) -> float:
    """Trains a Deep Neural Network.

    Note that this function will be executed by
    ``num_nodes * gpus_per_node`` processes/tasks. Those variables are
    set in the Hydra launcher configuration.

    Trains (or resumes training) the model, saves a checkpoint and
    returns the final validation loss.

    Args:
        trainer: See :class:`~lightning.pytorch.Trainer`.
        logger: See\
            :class:`~lightning.pytorch.loggers.wandb.WandbLogger`.
        datamodule: See :class:`.BaseDataModule`.
        litmodule: See :class:`.BaseLitModule`.
        config: See :paramref:`~.DeepLearningSubtaskConfig`.

    Returns:
        The final validation loss.
    """
    launcher_config = get_launcher_config()
    logger, trainer = instantiate_logger_and_trainer(
        config,
        launcher_config,
    )
    """TODO: Add logic for HPO"""
    set_batch_size_and_num_workers(
        config,
        trainer,
        datamodule,
    )
    ckpt_path = set_checkpoint_path(config, trainer)
    trainer.fit(model=litmodule, datamodule=datamodule, ckpt_path=ckpt_path)
    """TODO: Add logic for HPO
    trainer.save_checkpoint(filepath=config.model_load_path)
    """
    return trainer.validate(model=litmodule, datamodule=datamodule)[0][
        "val/loss"
    ]
