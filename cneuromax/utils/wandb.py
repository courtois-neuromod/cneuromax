""":mod:`wandb` utilities."""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cneuromax.utils.hydra import fs_builds


def login_wandb() -> None:
    """Logs in to W&B using the key stored in ``WANDB_KEY.txt``."""
    wandb_key_path = Path(
        str(os.environ.get("CNEUROMAX_PATH")) + "/WANDB_KEY.txt",
    )
    if wandb_key_path.exists():
        with wandb_key_path.open("r") as f:
            key = f.read().strip()
        wandb.login(key=key)
    else:
        logging.info(
            "W&B key not found, proceeding without. You can retrieve your key "
            "from `https://wandb.ai/settings` and store it in a file named "
            "`WANDB_KEY.txt` in the root directory of the project. Discard "
            "this message if you meant not to use W&B.",
        )


def store_logger_configs(cs: ConfigStore, clb: Callable[..., Any]) -> None:
    """Stores :mod:`hydra-core` ``logger`` group configs.

    Config names: ``wandb``, ``wandb_simexp``.

    Args:
        cs: See :paramref:`~.store_project_configs.cs`.
        clb: :mod:`wandb` initialization callable.
    """
    base_args: dict[str, Any] = {  # `fs_builds`` does not like dict[str, str]
        "name": "{task_run_dir}".split("/")[1],
        "save_dir": "${data_dir}",
        "project": "{task_run_dir}".split("/")[0],
    }

    cs.store(
        group="logger",
        name="wandb",
        node=fs_builds(clb, **base_args, entity=MISSING),
    )
    cs.store(
        group="logger",
        name="wandb_simexp",
        node=fs_builds(clb, **base_args, entity="cneuroml"),
    )
