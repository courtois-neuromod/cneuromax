r""":mod:`cneuromax`\-wide :mod:`hydra-core` config storing."""
from collections.abc import Callable
from typing import Any

from hydra_zen import ZenStore
from omegaconf import MISSING

from cneuromax.utils.hydra_zen import pfs_builds


def store_wandb_logger_configs(
    store: ZenStore,
    clb: Callable[..., Any],
    project: str,
) -> None:
    """Stores :mod:`hydra-core` ``logger`` group configs.

    Config names: ``wandb``, ``wandb_simexp``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        clb: :mod:`wandb` initialization callable.
        project: The :mod:`wandb` project name.
    """
    base_args: dict[str, Any] = {  # `fs_builds`` does not like dict[str, str]
        "name": "${task}",
        "save_dir": "${config.data_dir}",
        "project": project,
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
