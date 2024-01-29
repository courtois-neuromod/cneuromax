""":mod:`wandb` utilities."""

import logging
import os
from pathlib import Path

import wandb


def login_wandb() -> None:
    """Logs in to W&B using the key stored in ``WANDB_KEY.txt``."""
    wandb_key_path = Path(
        str(os.environ.get("CNEUROMAX_PATH")) + "/WANDB_KEY.txt",
    )
    if wandb_key_path.exists():
        with wandb_key_path.open(mode="r") as f:
            key = f.read().strip()
        wandb.login(key=key)
    else:
        logging.info(
            "W&B key not found, proceeding without. You can retrieve your key "
            "from `https://wandb.ai/settings` and store it in a file named "
            "`WANDB_KEY.txt` in the root directory of the project. Discard "
            "this message if you meant not to use W&B.",
        )
