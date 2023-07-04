"""."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import wandb
from cneuromax.deeplearning.common.logger import store as logger_store
from cneuromax.deeplearning.common.lrscheduler import (
    store as lrscheduler_store,
)
from cneuromax.deeplearning.common.optimizer import store as optimizer_store
from cneuromax.deeplearning.common.trainer import store as trainer_store
from cneuromax.deeplearning.fitter import Fitter

cs = ConfigStore.instance()
logger_store(cs)
lrscheduler_store(cs)
optimizer_store(cs)
trainer_store(cs)


@dataclass
class Config:
    """."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            {"logger": "wandb"},
            {"litmodule.optimizer_partial": "adamw"},
            {"litmodule.lrscheduler_partial": "none"},
            {"trainer": "base"},
            {"experiment": None},
            ["_self_"],
        ],
    )


cs.store(name="config", node=Config)


@hydra.main(
    version_base=None,
    config_name="config",
)
def run(config: DictConfig) -> float:
    """.

    Args:
        config (DictConfig): [description]

    Returns:
        The validation loss.
    """
    fitter = Fitter(config)

    return fitter.fit()


if __name__ == "__main__":
    # Retrieve the W&B key.
    with Path(str(os.environ.get("CNEUROMAX_PATH")) + "/WANDB_KEY.txt").open(
        "r",
    ) as f:
        key = f.read().strip()

    # Login to W&B.
    wandb.login(key=key)

    # Fit/Test.
    out = run()

    # If the main function returns a configuation, save it.
    if out:
        OmegaConf.save(out, "out.yaml")


# @hydra.main(config_path=None, config_name="config", version_base=None)
# def my_app(cfg: DictConfig) -> None:
#     """."""
#     # print(cfg)
#     dm = instantiate(cfg.dm)


# if __name__ == "__main__":
#     store.add_to_hydra_store()
#     cs = ConfigStore.instance()
#     cs.store(
#         name="config",
#         node=make_config(
#             hydra_defaults=[
#                 {"dm": "default"},
#                 "_self_",
#             ],
#         ),
#     )
#     my_app()
