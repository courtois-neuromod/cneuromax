"""."""

from dataclasses import dataclass


@dataclass
class BaseOptimizerConfig:
    """.

    Attributes:
        _target_: Optimizer class needs to be specified in child class.
        _partial_: NNModule parameters required for instantiation.
        lr: Learning rate.
        weight_decay: L2 penalty.
    """

    _target_: str = "???"
    _partial_: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-2
