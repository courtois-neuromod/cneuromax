"""Fitting function for Deep Learning."""

from cneuromax.fitting.deeplearning.config import (
    DeepLearningFittingHydraConfig,
)
from cneuromax.fitting.deeplearning.utils.lightning import (
    instantiate_lightning_objects,
    set_batch_size_and_num_workers,
    set_checkpoint_path,
)
from cneuromax.utils.hydra import get_launcher_config


def fit(config: DeepLearningFittingHydraConfig) -> float:
    """Trains a Deep Learning model.

    This function is the main entry point of the Deep Learning module.
    It acts as an interface between :mod:`hydra-core` (configuration +
    launcher + sweeper) and :mod:`lightning` (trainer + logger +
    modules).

    Note that this function will be executed by
    ``num_nodes * gpus_per_node`` processes/tasks. Those variables are
    set in the Hydra launcher configuration.

    Trains (or resumes training) the model, saves a checkpoint and
    returns the final validation loss.

    Args:
        config: The run's :mod:`hydra-core` structured config.

    Returns:
        The final validation loss.
    """
    launcher_config = get_launcher_config()
    logger, trainer, datamodule, litmodule = instantiate_lightning_objects(
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
