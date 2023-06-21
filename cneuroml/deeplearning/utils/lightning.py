"""PyTorch Lightning utilities for the fitting process."""

import logging
import time

import lightning.pytorch as pl
import numpy as np
from hydra.utils import instantiate
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from omegaconf import DictConfig


def find_good_batch_size(config: DictConfig) -> int:
    """Finds an appropriate ``batch_size`` parameter.

    This functionality makes the following, not always correct, but
    generally reasonable assumptions:
    - As long as the ``batch_size / dataset_size`` ratio remains small
    (e.g. < 0.01 so as to benefit from the stochasticity of gradient
    updates), running the same number of gradient updates with a larger
    batch size will yield faster training than running the same number
    of gradient updates with a smaller batch size.
    - Loading data from disk to RAM is a larger bottleneck than loading
    data from RAM to GPU VRAM.
    - If you are training on multiple GPUs, each GPU has roughly the
    same amount of VRAM.

    Args:
        config: Task configuration.

    Returns:
        batch_size: An estimated proper batch size.
    """
    logging.info("Finding good `batch_size` parameter...")

    # Instantiate a temporary Lightning Module.
    litmodule = instantiate(config.litmodule)

    # Instantiate a tempory Datamodule.
    datamodule = instantiate(config.datamodule)
    datamodule.num_workers = config.cpus_per_task

    # Instantiate a temporary Trainer running on a single GPU.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=-1,
    )

    # Instantiate a batch size Tuner.
    tuner = pl.tuner.tuning.Tuner(trainer=trainer)

    # Find the maximum batch size that fits on the GPU.
    tuner.scale_batch_size(
        model=litmodule,
        datamodule=datamodule,
        mode="binsearch",
    )  # value is stored in `datamodule.batch_size`.

    # Return 90% of maximum batch size to account for fluctuations.
    return int(datamodule.batch_size * 0.9)


def find_good_num_workers(
    config: DictConfig,
    batch_size: int,
    max_num_data_passes: int = 500,
) -> int:
    """Finds an appropriate `num_workers` parameter.

    Args:
        config: The task configuration.
        batch_size: The estimated appropriate batch size.
        max_num_data_passes: Maximum number of data passes to iterate
            through.

    Returns:
        num_workers: An estimated proper number of workers.
    """
    logging.info("Finding good `num_workers` parameter...")

    # Initialize a list to store the time taken for each `num_workers`.
    times = []
    # Loop over all reasonable `num_workers` values.
    for num_workers in range(1, config.cpus_per_task):
        # Instantiate a temporary datamodule.
        datamodule = instantiate(config.datamodule)
        datamodule.batch_size = batch_size
        datamodule.num_workers = num_workers
        # Start a timer.
        start_time = time.time()
        # Iterate through the dataloader `max_num_data_passes` times.
        num_data_passes = 0
        while num_data_passes < max_num_data_passes:
            for _ in datamodule.train_dataloader():
                num_data_passes += 1
                if num_data_passes == max_num_data_passes:
                    break
        # Stop the timer and store the time taken.
        times.append(time.time() - start_time)
        logging.info(
            f"num_workers: {num_workers}, time taken: {times[-1]}",
        )

    # Find the `num_workers` value that took the least amount of time.
    best_time = int(np.argmin(times))
    logging.info(f"Best `num_workers` parameter: {best_time + 1}.")
    return best_time + 1


class InitOptimParamsCheckpointConnector(_CheckpointConnector):
    """Initialized optimizer parameters Lightning checkpoint connector.

    Makes use of the newly initialized optimizers' parameters rather
    than the saved parameters. Useful for resuming training with
    different optimizer parameters (e.g. with the PBT Hydra sweeper).

    Attributes:
        trainer: The Lightning Trainer.
    """

    def restore_optimizers(self: "InitOptimParamsCheckpointConnector") -> None:
        """Restore optimizers w/ initialized optimizers'parameters."""
        # Retrieve the initialized optimizers' parameters.
        init_optims_param_groups = []
        for optimizer in self.trainer.strategy.optimizers:
            init_optims_param_groups.append(optimizer.param_groups)

        # Restore the optimizers through the checkpoint.
        super().restore_optimizers()

        # Place the initialized optimizers' parameters into the newly
        # instantiated optimizers.
        for optimizer, init_optim_param_group in zip(
            self.trainer.strategy.optimizers,
            init_optims_param_groups,
            strict=True,
        ):
            for optim_param_group_el, init_optim_param_group_el in zip(
                optimizer.param_groups,
                init_optim_param_group,
                strict=True,
            ):
                for param_group_key in optim_param_group_el:
                    if param_group_key != "params":
                        optim_param_group_el[
                            param_group_key
                        ] = init_optim_param_group_el[param_group_key]
