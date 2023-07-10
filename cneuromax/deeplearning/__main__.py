"""."""

import os
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_name="dlfitter",
    config_path="common",
)
def run(config: DictConfig) -> float:
    """.

    Args:
        config (DictConfig): .

    Returns:
        The validation loss.
    """
    from cneuromax.deeplearning.common import DeepLearningFitter

    fitter = DeepLearningFitter(config)
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
