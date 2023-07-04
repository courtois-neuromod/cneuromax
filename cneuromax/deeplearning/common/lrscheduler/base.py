"""."""

from dataclasses import dataclass


@dataclass
class BaseLRSchedulerConfig:
    """.

    Attributes:
        _partial_: Partial instantiation. Optimizer object required for
            full instantiation.
        last_epoch: Index of the last epoch when resuming training.
    """

    _partial_: bool = True
    last_epoch: int = -1
