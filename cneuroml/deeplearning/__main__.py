"""."""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from cneuroml.deeplearning.fitter import Fitter


@hydra.main(
    version_base=None,
    config_path=".",
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
    with Path(str(os.environ.get("CNEUROML_PATH")) + "/WANDB_KEY.txt").open(
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
