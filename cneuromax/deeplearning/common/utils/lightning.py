"""PyTorch Lightning utilities for the fitting process."""

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from lightning.pytorch.tuner.tuning import Tuner
from omegaconf import DictConfig

if TYPE_CHECKING:
    from cneuromax.deeplearning.common.datamodule import BaseDataModule
    from cneuromax.deeplearning.common.litmodule import BaseLitModule


def find_good_per_device_batch_size(config: DictConfig) -> int:
    """Finds an appropriate ``per_device_batch_size`` parameter.

    This functionality makes the following, not always correct, but
    generally reasonable assumptions:
    - As long as the ``total_batch_size / dataset_size`` ratio remains
    small (e.g. ``< 0.01`` so as to benefit from the stochasticity of
    gradient updates), running the same number of gradient updates with
    a larger batch size will yield faster training than running the same
    number of gradient updates with a smaller batch size.
    - Loading data from disk to RAM is a larger bottleneck than loading
    data from RAM to GPU VRAM.
    - If you are training on multiple GPUs, each GPU has roughly the
    same amount of VRAM.

    Args:
        config: The global Hydra configuration.

    Returns:
        per_device_batch_size: The estimated proper batch size per
            device.
    """
    logging.info("Finding good `batch_size` parameter...")

    # Instantiate a temporary LitModule.
    litmodule: BaseLitModule = instantiate(config.litmodule)

    # Instantiate a temporary Datamodule.
    datamodule: BaseDataModule = instantiate(config.datamodule)
    datamodule.config.per_device_num_workers = config.num_cpus_per_task

    # Instantiate a temporary Trainer running on a single GPU.
    trainer: Trainer = Trainer(
        accelerator=config.device,
        devices=1,
        max_epochs=-1,
    )

    # Instantiate a batch size Tuner.
    tuner: Tuner = Tuner(trainer=trainer)

    # Find the maximum batch size that fits on the GPU.
    per_device_batch_size: int | None = tuner.scale_batch_size(
        model=litmodule,
        datamodule=datamodule,
        mode="binsearch",
    )

    if per_device_batch_size is None:
        raise ValueError

    num_computing_devices: int
    if config.device == "gpu":
        num_computing_devices = config.num_nodes * config.num_gpus_per_node
    else:  # config.device == "cpu"
        num_computing_devices = config.num_nodes * config.num_tasks_per_node

    return min(
        # To account for GPU memory fluctuations
        int(per_device_batch_size * 0.9),
        # Ensure total batch_size is < 1% of the train dataloader size.
        len(datamodule.train_dataloader()) // (100 * num_computing_devices),
    )


def find_good_num_workers(
    config: DictConfig,
    per_device_batch_size: int,
    max_num_data_passes: int = 500,
) -> int:
    """Finds an appropriate `num_workers` parameter.

    Args:
        config: The global Hydra configuration.
        per_device_batch_size: .
        max_num_data_passes: Maximum number of data passes to iterate
            through.

    Returns:
        num_workers: An estimated proper number of workers.
    """
    logging.info("Finding good `num_workers` parameter...")

    if config.num_cpus_per_task == 1:
        logging.info("Only 1 worker available. Returning 0.")
        return 0

    # Initialize a list to store the time taken for each `num_workers`.
    times: list[float] = []
    # Loop over all reasonable `num_workers` values.
    for num_workers in range(0, config.num_cpus_per_task + 1):
        # Instantiate a temporary datamodule.
        datamodule: BaseDataModule = instantiate(config.datamodule)
        datamodule.config.per_device_batch_size = per_device_batch_size
        datamodule.config.per_device_num_workers = num_workers
        datamodule.prepare_data()
        datamodule.setup("fit")
        # Start a timer.
        start_time: float = time.time()
        # Iterate through the dataloader `max_num_data_passes` times.
        num_data_passes: int = 0
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

    # Find the ``num_workers`` value that took the least amount of time.
    best_time: int = int(np.argmin(times))
    logging.info(f"Best `num_workers` parameter: {best_time}.")
    return best_time


class InitOptimParamsCheckpointConnector(_CheckpointConnector):
    """Initialized optimizer parameters Lightning checkpoint connector.

    Makes use of the newly instantiated optimizers' hyper-parameters
    rather than the checkpointed hyper-parameters. Useful for resuming
    training with different optimizer hyper-parameters (e.g. with the
    PBT Hydra sweeper).
    """

    def restore_optimizers(self: "InitOptimParamsCheckpointConnector") -> None:
        """Restore optimizers but preserve newly instantiated params."""
        # Store the newly instantiated optimizers' parameters.
        new_inst_optim_param_groups: list[Any] = [
            optimizer.param_groups
            for optimizer in self.trainer.strategy.optimizers
        ]

        # Restore the optimizers through the checkpoint.
        super().restore_optimizers()

        # Place the Hydra instantiated optimizers' parameters back into
        # the newly restored optimizers.
        for optimizer, new_inst_optim_param_group in zip(
            self.trainer.strategy.optimizers,
            new_inst_optim_param_groups,
            strict=True,
        ):
            for optim_param_group_el, new_inst_optim_param_group_el in zip(
                optimizer.param_groups,
                new_inst_optim_param_group,
                strict=True,
            ):
                for param_group_key in optim_param_group_el:
                    # Skip the ``params`` key as it is not a
                    # hyper-parameter.
                    if param_group_key != "params":
                        optim_param_group_el[
                            param_group_key
                        ] = new_inst_optim_param_group_el[param_group_key]
