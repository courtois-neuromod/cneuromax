r""":mod:`cneuromax` `Hydra <https://hydra.cc>`_ config storing."""

from collections.abc import Callable
from typing import Any

from hydra_zen import ZenStore
from lightning.pytorch.loggers.wandb import WandbLogger

from cneuromax.utils.hydra_zen import pfs_builds


def store_wandb_logger_configs(
    store: ZenStore,
    clb: Callable[..., Any],
) -> None:
    """Stores `Hydra <https://hydra.cc>`_ ``logger`` group configs.

    Config names: ``wandb``, ``wandb_simexp``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        clb: `W&B <https://wandb.ai/>`_ initialization callable.
    """
    dir_key = "save_dir" if clb == WandbLogger else "dir"
    base_args: dict[str, Any] = {  # `fs_builds`` does not like dict[str, str]
        "name": "${task}/${hydra:job.override_dirname}",
        dir_key: "${hydra:sweep.dir}/${now:%Y-%m-%d-%H-%M-%S}",
        "project": "${project}",
    }
    store(
        pfs_builds(clb, **base_args),
        group="logger",
        name="wandb",
    )
    store(
        pfs_builds(clb, **base_args, entity="cneuroml"),
        group="logger",
        name="wandb_simexp",
    )
