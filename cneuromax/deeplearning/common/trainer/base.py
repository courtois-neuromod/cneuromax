"""."""

from dataclasses import dataclass
from typing import Annotated

from beartype.vale import Is
from hydra_zen import store


@store(name="base", group="trainer")
@dataclass
class BaseTrainerConfig:
    """.

    Attributes:
        _target_: Lightning Trainer.
        accelerator: Accelerator type.
        strategy: Training strategy.
        devices: Number of devices per node.
        num_nodes: .
        precision: Training precision.
        max_steps: Number of gradient updates.
        val_check_interval: How many gradient updates between validation
            checks.
        log_every_n_steps: How many gradient updates between logging
            actions.
    """

    _target_: str = "lightning.pytorch.trainer.Trainer"
    accelerator: Annotated[
        str,
        Is[lambda x: x in ("cpu", "gpu")],
    ] = "gpu"
    strategy: Annotated[
        str,
        Is[lambda x: x in ("auto", "ddp", "deepspeed")],
    ] = "auto"
    devices: Annotated[int, Is[lambda x: x >= 1]] | Annotated[
        int,
        Is[lambda x: x == -1],
    ] = 1
    num_nodes: Annotated[int, Is[lambda x: x >= 1]] = 1
    precision: Annotated[
        str,
        Is[lambda x: x in ("32", "16", "bf16")],
    ] = "32"
    max_steps: Annotated[int, Is[lambda x: x >= 1]] = int(1e3)
    val_check_interval: Annotated[
        int,
        Is[lambda x: x >= 1],
    ] = (
        int("${.max_steps}") // 10
    )
    log_every_n_steps: Annotated[
        int,
        Is[lambda x: x >= 1],
    ] = (
        int("${.max_steps}") // 10
    )
