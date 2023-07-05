"""."""

from dataclasses import dataclass

from hydra_zen import store

from cneuromax.deeplearning.common.trainer.base import BaseTrainerConfig


@store(name="ddp", group="trainer")
@dataclass
class DDPTrainer(BaseTrainerConfig):
    """.

    Attributes:
        strategy: Training strategy.
    """

    strategy = "ddp"
