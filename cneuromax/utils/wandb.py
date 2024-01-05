""":mod:`wandb` utilities."""
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import wandb
from hydra_zen import ZenStore
from omegaconf import MISSING

from cneuromax.utils.zen import fs_builds


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
    base_args: dict[str, Any] = {  # `fs_builds`` does not like dict[str, str]
        "name": "{output_dir}".split("/")[1],
        "save_dir": "${data_dir}",
        "project": "{output_dir}".split("/")[0],
    }
    store(
        fs_builds(clb, **base_args, entity=MISSING),
        group="logger",
        name="wandb",
    )
    store(
        fs_builds(clb, **base_args, entity="cneuroml"),
        group="logger",
        name="wandb_simexp",
    )
