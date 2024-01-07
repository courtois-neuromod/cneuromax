""":mod:`lightning` utilities."""
import copy
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Annotated as An

import numpy as np
import torch
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    SlurmLauncher,
)
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from lightning.pytorch.tuner.tuning import Tuner
from torch.distributed import ReduceOp
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

from cneuromax.fitting.config import (
    FittingSubtaskConfig,
)
from cneuromax.fitting.deeplearning.datamodule import BaseDataModule
from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.beartype import one_of
from cneuromax.utils.hydra import get_launcher_config
from cneuromax.utils.misc import get_path


def instantiate_trainer_and_logger(
    partial_trainer: partial[Trainer],
    partial_logger: partial[WandbLogger],
    device: An[str, one_of("cpu", "gpu")],
) -> tuple[Trainer, WandbLogger | None]:
    """Creates :mod:`lightning` instances.

    Args:
        partial_trainer: See :class:`~lightning.pytorch.Trainer`.
        partial_logger: See\
            :class:`~lightning.pytorch.loggers.wandb.WandbLogger`.
        device: See :paramref:`~.FittingSubtaskConfig.device`.

    Returns:
        * A :class:`~lightning.pytorch.Trainer` instance.
        * A :class:`~lightning.pytorch.loggers.wandb.WandbLogger`\
            instance or ``None``.
    """
    launcher_config = get_launcher_config()
    wandb_key_path = Path(
        str(os.environ.get("CNEUROMAX_PATH")) + "/WANDB_KEY.txt",
    )
    if wandb_key_path.exists():
        offline = launcher_config._target_ == get_path(  # noqa: SLF001
            SlurmLauncher,
        )
        logger = partial_logger(offline=offline)
    else:
        logging.info("W&B key not found. Logging disabled.")
        logger = None
    callbacks = None
    if launcher_config._target_ == get_path(SlurmLauncher):  # noqa: SLF001
        callbacks = [TriggerWandbSyncLightningCallback()]
    trainer = partial_trainer(
        devices=launcher_config.gpus_per_node or 1
        if device == "gpu"
        else launcher_config.tasks_per_node,
        logger=logger,
        callbacks=callbacks,
    )
    return trainer, logger


def set_batch_size_and_num_workers(
    trainer: Trainer,
    datamodule: BaseDataModule,
    litmodule: BaseLitModule,
    device: An[str, one_of("cpu", "gpu")],
    data_dir: str,
) -> None:
    """Sets attribute values for a :class:`~.BaseDataModule`.

    See :func:`find_good_per_device_batch_size` and
    :func:`find_good_per_device_num_workers` for more details on how
    these variables' values are determined.

    Args:
        trainer: See :class:`~lightning.pytorch.Trainer`.
        datamodule: See :class:`.BaseDataModule`.
        litmodule: See :class:`.BaseLitModule`.
        device: See :paramref:`~.FittingSubtaskConfig.device`.
        data_dir: See :paramref:`~.BaseSubtaskConfig.data_dir`.
    """
    proposed_per_device_batch_size = find_good_per_device_batch_size(
        litmodule=litmodule,
        datamodule=datamodule,
        device=device,
        data_dir=data_dir,
    )
    proposed_per_device_num_workers = find_good_per_device_num_workers(
        datamodule=datamodule,
        per_device_batch_size=proposed_per_device_batch_size,
    )
    per_device_batch_size = int(
        trainer.strategy.reduce(
            torch.tensor(proposed_per_device_batch_size),
            reduce_op=ReduceOp.MIN,  # type: ignore [arg-type]
        ),
    )
    per_device_num_workers = int(
        trainer.strategy.reduce(
            torch.tensor(proposed_per_device_num_workers),
            reduce_op=ReduceOp.MAX,  # type: ignore [arg-type]
        ),
    )
    datamodule.per_device_batch_size = per_device_batch_size
    datamodule.per_device_num_workers = per_device_num_workers


def find_good_per_device_batch_size(
    litmodule: BaseLitModule,
    datamodule: BaseDataModule,
    device: str,
    data_dir: str,
) -> int:
    """Probes a :attr:`~.BaseDataModule.per_device_batch_size` value.

    This functionality makes the following, not always correct, but
    generally reasonable assumptions:

    - As long as the ``total_batch_size / dataset_size`` ratio remains\
    small (e.g. ``< 0.01`` so as to benefit from the stochasticity of\
    gradient updates), running the same number of gradient updates with\
    a larger batch size will yield better training performance than\
    running the same number of gradient updates with a smaller batch\
    size.

    - Loading data from disk to RAM is a larger bottleneck than loading\
    data from RAM to GPU VRAM.

    - If you are training on multiple GPUs, each GPU has roughly the\
    same amount of VRAM.

    Args:
        litmodule: See :class:`.BaseLitModule`.
        datamodule: See :class:`.BaseDataModule`.
        device: See :paramref:`~.FittingSubtaskConfig.device`.
        data_dir: See :paramref:`~.BaseSubtaskConfig.data_dir`.

    Returns:
        A roughly optimal ``per_device_batch_size`` value.
    """
    litmodule_copy = copy.deepcopy(litmodule)
    datamodule_copy = copy.deepcopy(datamodule)
    launcher_config = get_launcher_config()
    datamodule_copy.per_device_num_workers = launcher_config.cpus_per_task or 1
    trainer = Trainer(
        accelerator=device,
        devices=1,
        max_epochs=-1,
        default_root_dir=data_dir + "/lightning/tuner/",
    )
    tuner = Tuner(trainer=trainer)
    logging.info("Finding good `batch_size` parameter...")
    per_device_batch_size = tuner.scale_batch_size(
        model=litmodule_copy,
        datamodule=datamodule_copy,
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
        len(datamodule_copy.train_dataloader())
        // (100 * num_computing_devices),
    )
    logging.info(f"Best `batch_size` parameter: {per_device_batch_size}.")
    return per_device_batch_size


def find_good_per_device_num_workers(
    datamodule: BaseDataModule,
    per_device_batch_size: int,
    max_num_data_passes: int = 100,
) -> int:
    """Probes a :attr:`~.BaseDataModule.per_device_num_workers` value.

    Iterates through a range of ``num_workers`` values and measures the
    time it takes to iterate through a fixed number of data passes;
    returning the value that yields the shortest time.

    Args:
        datamodule: See :class:`.BaseDataModule`.
        per_device_batch_size: The return value of\
            :func:`find_good_per_device_batch_size`.
        max_num_data_passes: Maximum number of data passes to iterate\
            through.

    Returns:
        A roughly optimal ``per_device_num_workers`` value.
    """
    launcher_config = get_launcher_config()
    logging.info("Finding good `num_workers` parameter...")
    if launcher_config.cpus_per_task in [None, 1]:
        logging.info("Only 1 worker available/provided. Returning 0.")
        return 0
    times = []
    for num_workers in range(launcher_config.cpus_per_task or 1 + 1):
        datamodule_copy = copy.deepcopy(datamodule)
        datamodule_copy.per_device_batch_size = per_device_batch_size
        datamodule_copy.per_device_num_workers = num_workers
        datamodule_copy.prepare_data()
        datamodule_copy.setup("fit")
        start_time = time.time()
        num_data_passes = 0
        while num_data_passes < max_num_data_passes:
            for _ in datamodule_copy.train_dataloader():
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


def set_checkpoint_path(
    trainer: Trainer,  # noqa: ARG001
    config: FittingSubtaskConfig,  # noqa: ARG001
) -> str | None:
    """Sets the path to the checkpoint to resume training from.

    TODO: Implement.

    Args:
        config: See :paramref:`~.FittingSubtaskConfig`.
        trainer: See :class:`~lightning.pytorch.Trainer`.

    Returns:
        The path to the checkpoint to resume training from.
    """
    return None


class InitOptimParamsCheckpointConnector(_CheckpointConnector):
    """Tweaked :mod:`lightning` checkpoint connector.

    Allows to make use of the instantiated optimizers'
    hyper-parameters rather than the checkpointed hyper-parameters.
    For use when resuming training with different optimizer
    hyper-parameters (e.g. with a PBT :mod:`hydra-core` Sweeper).
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
