"""."""

from dataclasses import dataclass


@dataclass
class BaseLRSchedulerConfig:
    """.

    Attributes:
        _target_: LRScheduler class needs to be specified in child
            class.
        _partial_: Optimizer object required for full instantiation.
        last_epoch: Index of the last epoch when resuming training.
    """

    _target_: str = "???"
    _partial_: bool = True
    last_epoch: int = -1
