"""CNeuroMax code base."""

import logging
import os
import warnings
from pathlib import Path

import wandb
from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))

# `beartype_this_package` needs to be called before importing any other
# modules from this package.
from cneuromax.config import (  # noqa: E402
    BaseHydraConfig,
    process_config,
    store_task_configs,
)

warnings.filterwarnings("ignore", module="beartype")
warnings.filterwarnings("ignore", module="lightning")

__all__ = [
    "BaseHydraConfig",
    "process_config",
    "store_task_configs",
    "login_wandb",
]


def login_wandb() -> None:
    """Logs in to W&B using the key stored in `WANDB_KEY.txt`."""
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
