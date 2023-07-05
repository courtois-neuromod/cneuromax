"""."""

from dataclasses import dataclass

from hydra_zen import store

from cneuromax.deeplearning.common.trainer.base import BaseTrainerConfig


@store(name="cpu", group="trainer")
@dataclass
class CPUTrainer(BaseTrainerConfig):
    """.

    Attributes:
        accelerator: Accelerator type.
    """

    accelerator = "cpu"
