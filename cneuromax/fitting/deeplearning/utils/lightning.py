"""Lightning utilities."""

import copy
import logging
import time

import numpy as np
from hydra.utils import instantiate
from hydra_plugins.hydra_submitit_launcher.config import (
    LocalQueueConf,
    SlurmQueueConf,
)
from lightning.pytorch import Trainer
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from lightning.pytorch.tuner.tuning import Tuner

from cneuromax.fitting.deeplearning.datamodule import BaseDataModule
from cneuromax.fitting.deeplearning.fitter import (
    DeepLearningFitterHydraConfig,
)
from cneuromax.fitting.deeplearning.litmodule import BaseLitModule


def find_good_per_device_batch_size(
    config: DeepLearningFitterHydraConfig,
    launcher_config: LocalQueueConf | SlurmQueueConf,
) -> int:
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
        config: .
        launcher_config: The Hydra launcher configuration.

    Returns:
        per_device_batch_size: The estimated proper batch size per
            device.
    """
    logging.info("Finding good `batch_size` parameter...")
    litmodule: BaseLitModule = instantiate(config.litmodule)
    datamodule: BaseDataModule = instantiate(config.datamodule)
    datamodule.per_device_num_workers = launcher_config.cpus_per_task or 1
    trainer = Trainer(
        accelerator=config.device,
        devices=1,
        max_epochs=-1,
        default_root_dir=config.data_dir + "/lightning/tuner/",
    )
    tuner = Tuner(trainer=trainer)
    per_device_batch_size = tuner.scale_batch_size(
        model=litmodule,
        datamodule=datamodule,
        mode="binsearch",
        batch_arg_name="per_device_batch_size",
    )
    if per_device_batch_size is None:
        raise ValueError  # Won't happen according to Lightning source code.
    num_computing_devices = launcher_config.nodes * (
        launcher_config.gpus_per_node or 1
        if config.device == "gpu"
        else launcher_config.tasks_per_node
    )
    per_device_batch_size: int = min(
        # Account for GPU memory discrepancies & ensure total batch_size
        # is < 1% of the train dataloader size.
        int(per_device_batch_size * 0.9),
        len(datamodule.train_dataloader()) // (100 * num_computing_devices),
    )
    logging.info(f"Best `batch_size` parameter: {per_device_batch_size}.")
    return per_device_batch_size


def find_good_num_workers(
    config: DeepLearningFitterHydraConfig,
    launcher_config: LocalQueueConf | SlurmQueueConf,
    per_device_batch_size: int,
    max_num_data_passes: int = 100,
) -> int:
    """Finds an appropriate `num_workers` parameter.

    This function makes use of the ``per_device_batch_size`` parameter
    found by the ``find_good_per_device_batch_size`` function in order
    to find an appropriate ``num_workers`` parameter.
    It does so by iterating through a range of ``num_workers`` values
    and measuring the time it takes to iterate through a fixed number of
    data passes; picking the ``num_workers`` value that yields the
    shortest time.

    Args:
        config: .
        launcher_config: The Hydra launcher configuration.
        per_device_batch_size: .
        max_num_data_passes: Maximum number of data passes to iterate
            through.

    Returns:
        num_workers: An estimated proper number of workers.
    """
    logging.info("Finding good `num_workers` parameter...")
    if launcher_config.cpus_per_task in [None, 1]:
        logging.info("Only 1 worker available/provided. Returning 0.")
        return 0
    times = []
    for num_workers in range(launcher_config.cpus_per_task or 1 + 1):
        datamodule: BaseDataModule = instantiate(config.datamodule)
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
    """Initialized optimizer parameters Lightning checkpoint connector.

    Makes use of the newly instantiated optimizers' hyper-parameters
    rather than the checkpointed hyper-parameters. For use when resuming
    training with different optimizer hyper-parameters (e.g. with the
    PBT Hydra sweeper).
    """

    def restore_optimizers(self: "InitOptimParamsCheckpointConnector") -> None:
        """Preserves newly instantiated parameters."""
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
                    # Skip the `params`` key as it is not a HP.
                    if ckpt_optim_param_group_key != "params":
                        # Place the new Hydra instantiated optimizers'
                        # HPs back into the restored optimizers.
                        ckpt_optim_param_group[
                            ckpt_optim_param_group_key
                        ] = new_optim_param_group[ckpt_optim_param_group_key]
