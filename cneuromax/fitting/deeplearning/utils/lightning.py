""":mod:`lightning` utilities."""

import contextlib
import copy
import logging
import math
import os
import sys
import time
from functools import partial
from typing import Annotated as An
from typing import Any

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BatchSizeFinder, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from omegaconf import OmegaConf
from torch.distributed import ReduceOp

from cneuromax.fitting.deeplearning.datamodule import BaseDataModule
from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.fitting.utils.hydra import get_launcher_config
from cneuromax.utils.beartype import one_of
from cneuromax.utils.misc import can_connect_to_internet


def instantiate_trainer(
    trainer_partial: partial[Trainer],
    logger_partial: partial[WandbLogger],
    device: An[str, one_of("cpu", "gpu")],
    output_dir: str,
    save_every_n_train_steps: int | None,
) -> Trainer:
    """Instantiates a :class:`~lightning.pytorch.Trainer`.

    Args:
        trainer_partial: See :class:`~lightning.pytorch.Trainer`.
        logger_partial: See
            :class:`~lightning.pytorch.loggers.wandb.WandbLogger`.
        device: See :paramref:`~.FittingSubtaskConfig.device`.
        output_dir: See :paramref:`~.BaseSubtaskConfig.output_dir`.
        save_every_n_train_steps: See
            :paramref:`~.DeepLearningSubtaskConfig.save_every_n_train_steps`.

    Returns:
        A :class:`~lightning.pytorch.Trainer` instance.
    """
    launcher_config = get_launcher_config()
    # Retrieve the `Trainer` callbacks specified in the config
    callbacks: list[Any] = trainer_partial.keywords["callbacks"] or []
    # Check internet connection
    offline = (
        logger_partial.keywords["offline"] or not can_connect_to_internet()
    )
    if offline:
        os.environ["WANDB_DISABLE_SERVICE"] = "True"
    # Adds a callback to save the state of training (not just the model
    # despite the name) at the end of every validation epoch.
    callbacks.append(
        ModelCheckpoint(
            dirpath=trainer_partial.keywords["default_root_dir"],
            monitor="val/loss",
            save_last=True,
            save_top_k=1,
            every_n_train_steps=save_every_n_train_steps,
        ),
    )
    # Instantiate the :class:`WandbLogger`.`
    logger = logger_partial(offline=offline)
    # Feed the :mod:`hydra` config to :mod:`wandb`.
    logger.experiment.config.update(
        OmegaConf.to_container(
            OmegaConf.load(f"{output_dir}/.hydra/config.yaml"),
            resolve=True,
            throw_on_missing=True,
        ),
    )
    # Instantiate the :class:`~lightning.pytorch.Trainer`.
    return trainer_partial(
        devices=(
            launcher_config.gpus_per_node or 1
            if device == "gpu"
            else launcher_config.tasks_per_node
        ),
        logger=logger,
        callbacks=callbacks,
    )


def set_batch_size_and_num_workers(
    trainer: Trainer,
    datamodule: BaseDataModule,
    litmodule: BaseLitModule,
    device: An[str, one_of("cpu", "gpu")],
    output_dir: str,
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
        output_dir: See :paramref:`~.BaseSubtaskConfig.output_dir`.
    """
    if not datamodule.config.fixed_per_device_batch_size:
        proposed_per_device_batch_size = find_good_per_device_batch_size(
            litmodule=litmodule,
            datamodule=datamodule,
            device=device,
            device_ids=trainer.device_ids,
            output_dir=output_dir,
        )
        per_device_batch_size = int(
            trainer.strategy.reduce(
                torch.tensor(proposed_per_device_batch_size),
                reduce_op=ReduceOp.MIN,  # type: ignore [arg-type]
            ),
        )
    else:
        per_device_batch_size = datamodule.config.fixed_per_device_batch_size

    if not datamodule.config.fixed_per_device_num_workers:
        proposed_per_device_num_workers = find_good_per_device_num_workers(
            datamodule=datamodule,
            per_device_batch_size=per_device_batch_size,
        )
        per_device_num_workers = int(
            trainer.strategy.reduce(
                torch.tensor(proposed_per_device_num_workers),
                reduce_op=ReduceOp.MAX,  # type: ignore [arg-type]
            ),
        )
    else:
        per_device_num_workers = datamodule.config.fixed_per_device_num_workers

    datamodule.per_device_batch_size = per_device_batch_size
    datamodule.per_device_num_workers = per_device_num_workers


def find_good_per_device_batch_size(
    litmodule: BaseLitModule,
    datamodule: BaseDataModule,
    device: str,
    device_ids: list[int],
    output_dir: str,
) -> int:
    """Probes a :attr:`~.BaseDataModule.per_device_batch_size` value.

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
        litmodule: See :class:`.BaseLitModule`.
        datamodule: See :class:`.BaseDataModule`.
        device: See :paramref:`~.FittingSubtaskConfig.device`.
        device_ids: See :class:`~lightning.pytorch.Trainer.device_ids`.
        output_dir: See :paramref:`~.BaseSubtaskConfig.output_dir`.

    Returns:
        A roughly optimal ``per_device_batch_size`` value.
    """
    launcher_config = get_launcher_config()
    datamodule_copy = copy.deepcopy(datamodule)
    datamodule_copy.prepare_data()
    datamodule_copy.setup("fit")
    # (since the default `datamodule` `batch_size` is 1)
    dataset_len = len(datamodule_copy.train_dataloader())
    num_computing_devices = launcher_config.nodes * (
        launcher_config.gpus_per_node or 1
        if device == "gpu"
        else launcher_config.tasks_per_node
    )
    per_device_batch_size: int | None
    # Ensures total batch_size is < 1% of the train dataloader size.
    max_per_device_batch_size = dataset_len // (100 * num_computing_devices)
    # If a maximum batch size was specified, use the smaller of the two.
    if datamodule.config.max_per_device_batch_size:
        max_per_device_batch_size = min(
            max_per_device_batch_size,
            datamodule.config.max_per_device_batch_size,
        )
    if device == "cpu":
        # Only running batch size finding on GPU: reaching out of GPU
        # memory errors does not freeze the system wheras reaching out
        # of CPU memory errors does.
        per_device_batch_size = max_per_device_batch_size
    else:
        litmodule_copy = copy.deepcopy(litmodule)
        # Speeds up the batch size search by removing the validation
        # epoch end method, which is independent of the batch size.
        litmodule_copy.on_validation_epoch_end = None  # type: ignore[assignment,method-assign]
        # Speeds up the batch size search by using a reasonable number
        # of workers for the search.
        if launcher_config.cpus_per_task:
            datamodule_copy.per_device_num_workers = (
                launcher_config.cpus_per_task
            )
        batch_size_finder = BatchSizeFinder(
            mode="power",
            batch_arg_name="per_device_batch_size",
            max_trials=int(math.log2(max_per_device_batch_size)),
        )
        # Stops the `fit` method after the batch size has been found.
        batch_size_finder._early_exit = True  # noqa: SLF001
        trainer = Trainer(
            accelerator=device,
            devices=[device_ids[0]],  # The first available device.
            default_root_dir=output_dir + "/lightning/tuner/",
            callbacks=[batch_size_finder],
        )
        logging.info("Finding good `batch_size` parameter...")
        per_device_batch_size = None
        # Prevents the `fit` method from raising a `KeyError`, see:
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18114
        with contextlib.suppress(KeyError):
            trainer.fit(model=litmodule_copy, datamodule=datamodule_copy)
        # If any OOM occurs the value is stored in
        # `per_device_batch_size` else in `optimal_batch_size`.
        optimal_per_device_batch_size = max(
            datamodule_copy.per_device_batch_size,
            batch_size_finder.optimal_batch_size or 1,
        )
        per_device_batch_size = min(
            optimal_per_device_batch_size,
            max_per_device_batch_size,
        )
    if per_device_batch_size == 0:
        per_device_batch_size = 1
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
        per_device_batch_size: The return value of
            :func:`find_good_per_device_batch_size`.
        max_num_data_passes: Maximum number of data passes to iterate
            through.

    Returns:
        A roughly optimal ``per_device_num_workers`` value.
    """
    launcher_config = get_launcher_config()
    logging.info("Finding good `num_workers` parameter...")
    if launcher_config.cpus_per_task in [None, 1]:
        logging.info("Only 1 worker available/provided. Returning 0.")
        return 0
    # Static type checking purposes, already narrowed down to `int`
    # through the `if` statement above.
    assert launcher_config.cpus_per_task  # noqa: S101
    times: list[float] = [
        sys.float_info.max for _ in range(launcher_config.cpus_per_task + 1)
    ]
    datamodule_copy = copy.deepcopy(datamodule)
    datamodule_copy.per_device_batch_size = per_device_batch_size
    datamodule_copy.prepare_data()
    datamodule_copy.setup("fit")
    for num_workers in range(launcher_config.cpus_per_task, -1, -1):
        datamodule_copy.per_device_num_workers = num_workers
        start_time = time.time()
        num_data_passes = 0
        while num_data_passes < max_num_data_passes:
            for _ in datamodule_copy.train_dataloader():
                num_data_passes += 1
                if num_data_passes == max_num_data_passes:
                    break
        times[num_workers] = time.time() - start_time
        logging.info(
            f"num_workers: {num_workers}, time taken: {times[num_workers]}",
        )
        # If the time taken is not decreasing, stop the search.
        if (
            # Not after the first iteration.
            num_workers != launcher_config.cpus_per_task  # noqa: PLR1714
            # Still want to attempt `num_workers` = 0.
            and num_workers != 1
            and times[num_workers + 1] <= times[num_workers]
        ):
            break
    best_time = int(np.argmin(times))
    logging.info(f"Best `num_workers` parameter: {best_time}.")
    return best_time


class InitOptimParamsCheckpointConnector(_CheckpointConnector):
    """Tweaked :mod:`lightning` checkpoint connector.

    Allows to make use of the instantiated optimizers'
    hyper-parameters rather than the checkpointed hyper-parameters.
    For use when resuming training with different optimizer
    hyper-parameters (e.g. with a PBT :mod:`hydra` Sweeper).
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
                        ckpt_optim_param_group[ckpt_optim_param_group_key] = (
                            new_optim_param_group[ckpt_optim_param_group_key]
                        )
