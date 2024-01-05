""":mod:`hydra-zen` utilities."""
from hydra_zen import ZenStore
from lightning.pytorch.trainer import Trainer
from torch.optim import SGD, Adam, AdamW
from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

from cneuromax.utils.zen import fs_builds, pfs_builds


def store_basic_optimizer_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/optimizer`` group configs.

    Config names: ``adam``, ``adamw``, ``sgd``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(pfs_builds(Adam), name="adam", group="litmodule/optimizer")
    store(pfs_builds(AdamW), name="adamw", group="litmodule/optimizer")
    store(pfs_builds(SGD), name="sgd", group="litmodule/optimizer")


def store_basic_scheduler_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/scheduler`` group configs.

    Config names: ``constant``, ``linear_warmup``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        pfs_builds(get_constant_schedule),
        name="constant",
        group="litmodule/scheduler",
    )
    store(
        pfs_builds(get_constant_schedule_with_warmup),
        name="linear_warmup",
        group="litmodule/scheduler",
    )


def store_basic_trainer_config(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``trainer`` group configs.

    Config name: ``base``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        fs_builds(
            Trainer,
            accelerator="${device}",
            default_root_dir="${subtask_run_dir}/lightning/",
        ),
        name="base",
        group="trainer",
    )
