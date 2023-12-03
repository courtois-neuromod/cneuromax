"""Fitting with Deep Learning (+ :mod:`hydra-core` config storing)."""

from hydra.core.config_store import ConfigStore
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.trainer import Trainer
from omegaconf import MISSING
from torch.optim import SGD, Adam, AdamW
from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

from cneuromax.fitting.deeplearning.config import (
    DeepLearningFittingHydraConfig,
)
from cneuromax.fitting.deeplearning.nnmodule import store_nnmodule_configs
from cneuromax.utils.hydra import fs_builds, pfs_builds

__all__ = [
    "DeepLearningFitter",
    "DeepLearningFitterHydraConfig",
    "store_deep_learning_configs",
]


def store_deep_learning_configs(cs: ConfigStore) -> None:
    """Store :mod:`hydra-core` Deep Learning configs.

    Args:
        cs: See :paramref:`cneuromax.config.store_task_configs.cs`.
    """
    store_logger_configs(cs)
    store_nnmodule_configs(cs)
    store_optimizer_configs(cs)
    store_scheduler_configs(cs)
    store_trainer_configs(cs)
    cs.store(name="dl_fitting", node=DeepLearningFittingHydraConfig)


def store_logger_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` ``logger`` group configs.

    Config names: ``wandb``, ``wandb_simexp``.

    Args:
        cs: See :paramref:`cneuromax.config.store_task_configs.cs`.
    """
    base_args = {
        "name": MISSING,
        "save_dir": "${data_dir}",
        "project": MISSING,
    }
    cs.store(
        group="logger",
        name="wandb",
        node=fs_builds(WandbLogger, **base_args, entity=MISSING),
    )

    cs.store(
        group="logger",
        name="wandb_simexp",
        node=fs_builds(WandbLogger, **base_args, entity="cneuroml"),
    )


def store_optimizer_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/optimizer`` group configs.

    Config names: ``adam``, ``adamw``, ``sgd``.

    Args:
        cs: See :paramref:`cneuromax.config.store_task_configs.cs`.
    """
    cs.store(group="litmodule/optimizer", name="adam", node=pfs_builds(Adam))
    cs.store(group="litmodule/optimizer", name="adamw", node=pfs_builds(AdamW))
    cs.store(group="litmodule/optimizer", name="sgd", node=pfs_builds(SGD))


def store_scheduler_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/scheduler`` group configs.

    Config names: ``constant``, ``linear_warmup``.

    Args:
        cs: See :paramref:`cneuromax.config.store_task_configs.cs`.
    """
    cs.store(
        group="litmodule/scheduler",
        name="constant",
        node=pfs_builds(get_constant_schedule),
    )
    cs.store(
        group="litmodule/scheduler",
        name="linear_warmup",
        node=pfs_builds(get_constant_schedule_with_warmup),
    )


def store_trainer_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` ``trainer`` group configs.

    Config name: ``base``.

    Args:
        cs: See :paramref:`cneuromax.config.store_task_configs.cs`.
    """
    cs.store(
        group="trainer",
        name="base",
        node=fs_builds(
            Trainer,
            accelerator="${device}",
            default_root_dir="${data_dir}/lightning/",
        ),
    )
