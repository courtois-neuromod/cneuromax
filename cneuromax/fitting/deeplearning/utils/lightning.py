""":mod:`lightning` utilities."""

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from hydra_plugins.hydra_submitit_launcher.config import (
    LocalQueueConf,
    SlurmQueueConf,
)
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    SlurmLauncher,
)
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from lightning.pytorch.tuner.tuning import Tuner
from torch.distributed import ReduceOp
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

from cneuromax.fitting.deeplearning.config import (
    DeepLearningFittingHydraConfig,
)
from cneuromax.fitting.deeplearning.datamodule import BaseDataModule
from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.hydra import get_launcher_config, get_path


def instantiate_lightning_objects(
    config: DeepLearningFittingHydraConfig,
    launcher_config: LocalQueueConf | SlurmQueueConf,
) -> tuple[Logger | None, Trainer, BaseDataModule, BaseLitModule]:
    """Creates several :mod:`lightning` objects based on the config.

    Args:
        config: See\
            :class:`cneuromax.fitting.deeplearning.config.DeepLearningFittingHydraConfig`.
        launcher_config: The :mod:`hydra-core` launcher configuration.

    Returns:
        The instantiated :mod:`lightning` objects.
    """
    logger: Logger | None
    if config.logger._target_ == get_path(WandbLogger):  # noqa: SLF001
        wandb_key_path = Path(
            str(os.environ.get("CNEUROMAX_PATH")) + "/WANDB_KEY.txt",
        )
        if wandb_key_path.exists():
            kwargs = {}
            if launcher_config._target_ == get_path(  # noqa: SLF001
                SlurmLauncher,
            ):
                kwargs["offline"] = True
            logger = instantiate(config.logger, **kwargs)
        else:
            logging.info("W&B key not found. Logging disabled.")
            logger = None
    else:
        logger = instantiate(config.logger)
    callbacks = None
    if launcher_config._target_ == get_path(SlurmLauncher):  # noqa: SLF001
        callbacks = [TriggerWandbSyncLightningCallback()]
    trainer: Trainer = instantiate(
        config=config.trainer,
        devices=launcher_config.gpus_per_node or 1
        if config.device == "gpu"
        else launcher_config.tasks_per_node,
        logger=logger,
        callbacks=callbacks,
    )
    datamodule: BaseDataModule = instantiate(config.datamodule)
    litmodule: BaseLitModule = instantiate(config.litmodule)
    return logger, trainer, datamodule, litmodule


def set_batch_size_and_num_workers(
    config: DeepLearningFittingHydraConfig,
    trainer: Trainer,
    datamodule: BaseDataModule,
) -> None:
    """Computes and sets the batch size and number of workers.

    If starting a new HPO run, finds and sets "good" ``batch_size`` and
    ``num_workers`` parameters.

    See the
    :func:`cneuromax.fitting.deeplearning.utils.lightning.find_good_batch_size`
    and
    :func:`cneuromax.fitting.deeplearning.utils.lightning.find_good_num_workers`
    functions documentation for more details.

    We make the assumption that if we are resuming from a checkpoint
    created while running hyper-parameter optimization, we are running
    on the same hardware configuration as was used to create the
    checkpoint. Therefore, we do not need to once again look for good
    ``batch_size`` and ``num_workers`` parameters.

    Args:
        config: See\
            :class:`cneuromax.fitting.deeplearning.config.DeepLearningFittingHydraConfig`.
        trainer: The :class:`lightning.pytorch.Trainer` instance used\
            for this fitting run.
        datamodule: The\
            :class:`cneuromax.fitting.deeplearning.datamodule.BaseDataModule`\
            instance used for this fitting run.
    """
    if not config.pbt_load_path:
        proposed_per_device_batch_size: int = find_good_per_device_batch_size(
            litmodule=instantiate(config.litmodule),
            datamodule=instantiate(config.datamodule),
            device=config.device,
            data_dir=config.data_dir,
        )
        proposed_per_device_num_workers: int = find_good_num_workers(
            datamodule_config=config.datamodule,
            per_device_batch_size=proposed_per_device_batch_size,
        )
    per_device_batch_size: int = int(
        trainer.strategy.reduce(
            torch.tensor(proposed_per_device_batch_size),
            reduce_op=ReduceOp.MIN,  # type: ignore [arg-type]
        ),
    )
    per_device_num_workers: int = int(
        trainer.strategy.reduce(
            torch.tensor(proposed_per_device_num_workers),
            reduce_op=ReduceOp.MAX,  # type: ignore [arg-type]
        ),
    )
    datamodule.per_device_batch_size = per_device_batch_size
    datamodule.per_device_num_workers = per_device_num_workers


def set_checkpoint_path(
    config: DeepLearningFittingHydraConfig,
    trainer: Trainer,
) -> str | None:
    """Sets the path to the checkpoint to resume training from.

    Three cases are considered:
    - if the :paramref:`DeepLearningFittingHydraConfig.load_path_pbt`
    parameter is set, we are resuming from a checkpoint created while
    running HPO. In this case, we set the checkpoint path to the value
    of :paramref:`DeepLearningFittingHydraConfig.load_path_hpo` and
    use a custom checkpoint connector to not override the new HPO
    config values.
    - if the :paramref:`DeepLearningFittingHydraConfig.load_path`
    parameter is set (but not
    :paramref:`DeepLearningFittingHydraConfig.load_path_hpo`), we are
    resuming from a regular checkpoint. In this case, we set the
    checkpoint path to the value of
    :paramref:`DeepLearningFittingHydraConfig.load_path`.
    - if neither
    :paramref:`DeepLearningFittingHydraConfig.load_path_hpo` nor
    :paramref:`DeepLearningFittingHydraConfig.load_path` are set, we
    are starting a new training run. In this case, we set the
    checkpoint path to `None`.

    Args:
        config: The config instance used for this fitting run.
        trainer: The :class:`lightning.pytorch.Trainer` instance used\
            for this fitting run.

    Returns:
        The path to the checkpoint to resume training from.
    """
    ckpt_path: str | None
    if config.pbt_load_path:
        ckpt_path = config.pbt_load_path
        # No choice but to access a private attribute.
        trainer._checkpoint_connector = (  # noqa: SLF001
            InitOptimParamsCheckpointConnector(
                trainer,
            )
        )
    elif config.model_load_path:
        ckpt_path = config.model_load_path
    else:
        ckpt_path = None

    return ckpt_path


def find_good_per_device_batch_size(
    litmodule: BaseLitModule,
    datamodule: BaseDataModule,
    device: str,
    data_dir: str,
) -> int:
    """Finds an appropriate ``per_device_batch_size`` parameter.

    This functionality makes the following, not always correct, but
    generally reasonable assumptions:
    - As long as the ``total_batch_size / dataset_size`` ratio remains
    small (e.g. ``< 0.01`` so as to benefit from the stochasticity of
    gradient updates), running the same number of gradient updates with
    a larger batch size will yield better training performance than
    running the same number of gradient updates with a smaller batch
    size.
    - Loading data from disk to RAM is a larger bottleneck than loading
    data from RAM to GPU VRAM.
    - If you are training on multiple GPUs, each GPU has roughly the
    same amount of VRAM.

    Args:
        litmodule: A temporary\
            :class:`cneuromax.fitting.deeplearning.litmodule.base.BaseLitModule`\
            instance with the same configuration as the\
            :class:`cneuromax.fitting.deeplearning.litmodule.base.BaseLitModule`\
            instance that will be trained.
        datamodule: A temporary\
            :class:`cneuromax.fitting.deeplearning.datamodule.base.BaseDataModule`\
            instance with the same configuration as the\
            :class:`cneuromax.fitting.deeplearning.datamodule.base.BaseDataModule`\
            instance that will be used for training.
        device: See\
            :paramref:`cneuromax.fitting.config.BaseFittingHydraConfig.device`.
        data_dir: See\
            :paramref:`cneuromax.fitting.config.BaseFittingHydraConfig.data_dir`.

    Returns:
        The estimated proper batch size per device.
    """
    launcher_config = get_launcher_config()
    datamodule.per_device_num_workers = launcher_config.cpus_per_task or 1
    trainer = Trainer(
        accelerator=device,
        devices=1,
        max_epochs=-1,
        default_root_dir=data_dir + "/lightning/tuner/",
    )
    tuner = Tuner(trainer=trainer)
    logging.info("Finding good `batch_size` parameter...")
    per_device_batch_size = tuner.scale_batch_size(
        model=litmodule,
        datamodule=datamodule,
        mode="binsearch",
        batch_arg_name="per_device_batch_size",
    )
    if per_device_batch_size is None:
        error_msg = (
            "Lightning's `scale_batch_size` method returned `None`. "
            "This is outside of the user's control, please try again."
        )
        raise ValueError(error_msg)
    num_computing_devices = launcher_config.nodes * (
        launcher_config.gpus_per_node or 1
        if device == "gpu"
        else launcher_config.tasks_per_node
    )
    per_device_batch_size: int = min(
        # Account for GPU memory discrepancies & ensure total batch size
        # is < 1% of the train dataloader size.
        int(per_device_batch_size * 0.9),
        len(datamodule.train_dataloader()) // (100 * num_computing_devices),
    )
    logging.info(f"Best `batch_size` parameter: {per_device_batch_size}.")
    return per_device_batch_size


def find_good_num_workers(
    # Type is implicit through :mod:`hydra-zen`.
    datamodule_config: Any,  # noqa: ANN401
    per_device_batch_size: int,
    max_num_data_passes: int = 100,
) -> int:
    """Finds an appropriate ``num_workers`` parameter.

    This function makes use of the ``per_device_batch_size`` parameter
    found by the ``find_good_per_device_batch_size`` function in order
    to find an appropriate ``num_workers`` parameter.
    It does so by iterating through a range of ``num_workers`` values
    and measuring the time it takes to iterate through a fixed number of
    data passes; picking the ``num_workers`` value that yields the
    shortest time.

    Args:
        datamodule_config: Implicit (generated by :mod:`hydra-zen`)\
            ``DataModuleHydraConfig`` instance.
        per_device_batch_size: The batch size returned by\
            :func:`find_good_per_device_batch_size`.
        max_num_data_passes: Maximum number of data passes to iterate\
            through (default: ``100``).

    Returns:
        The estimated proper number of workers.
    """
    launcher_config = get_launcher_config()
    logging.info("Finding good `num_workers` parameter...")
    if launcher_config.cpus_per_task in [None, 1]:
        logging.info("Only 1 worker available/provided. Returning 0.")
        return 0
    times = []
    for num_workers in range(launcher_config.cpus_per_task or 1 + 1):
        datamodule: BaseDataModule = instantiate(datamodule_config)
        datamodule.per_device_batch_size = per_device_batch_size
        datamodule.per_device_num_workers = num_workers
        datamodule.prepare_data()
        datamodule.setup("fit")
        start_time = time.time()
        num_data_passes = 0
        while num_data_passes < max_num_data_passes:
            for _ in datamodule.train_dataloader():
                num_data_passes += 1
                if num_data_passes == max_num_data_passes:
                    break
        times.append(time.time() - start_time)
        logging.info(
            f"num_workers: {num_workers}, time taken: {times[-1]}",
        )
    best_time = int(np.argmin(times))
    logging.info(f"Best `num_workers` parameter: {best_time}.")
    return best_time


class InitOptimParamsCheckpointConnector(_CheckpointConnector):
    """Tweaked ckpt connector to preserve newly instantiated parameters.

    Allows to make use of the newly instantiated optimizers'
    hyper-parameters rather than the checkpointed hyper-parameters.
    For use when resuming training with different optimizer
    hyper-parameters (e.g. with the PBT :mod:`hydra-core` sweeper).
    """

    def restore_optimizers(self: "InitOptimParamsCheckpointConnector") -> None:
        """Tweaked method to preserve newly instantiated parameters."""
        new_optims = copy.deepcopy(self.trainer.strategy.optimizers)
        super().restore_optimizers()
        for ckpt_optim, new_optim in zip(
            self.trainer.strategy.optimizers,
            new_optims,
            strict=True,
        ):
            for ckpt_optim_param_group, new_optim_param_group in zip(
                ckpt_optim.param_groups,
                new_optim.param_groups,
                strict=True,
            ):
                for ckpt_optim_param_group_key in ckpt_optim_param_group:
                    # Skip the `params` key as it is not a HP.
                    if ckpt_optim_param_group_key != "params":
                        # Place the new Hydra instantiated optimizers'
                        # HPs back into the restored optimizers.
                        ckpt_optim_param_group[
                            ckpt_optim_param_group_key
                        ] = new_optim_param_group[ckpt_optim_param_group_key]
