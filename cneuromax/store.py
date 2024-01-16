r""":mod:`cneuromax`\-wide :mod:`hydra-core` config storing."""
from collections.abc import Callable
from typing import Any

from hydra_zen import ZenStore
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import MISSING

from cneuromax.utils.hydra_zen import pfs_builds


def store_wandb_logger_configs(
    store: ZenStore,
    clb: Callable[..., Any],
) -> None:
    """Stores :mod:`hydra-core` ``logger`` group configs.

    Config names: ``wandb``, ``wandb_simexp``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        clb: :mod:`wandb` initialization callable.
    """
    dir_key = "save_dir" if isinstance(clb, WandbLogger) else "dir"
    base_args: dict[str, Any] = {  # `fs_builds`` does not like dict[str, str]
        "name": "${task}",
        dir_key: "${config.output_dir}",
        "project": "${project}",
    }
    store(
        pfs_builds(clb, **base_args, entity=MISSING),
        group="logger",
        name="wandb",
    )
    store(
        pfs_builds(clb, **base_args, entity="cneuroml"),
        group="logger",
        name="wandb_simexp",
    )
