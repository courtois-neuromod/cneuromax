"""."""

from dataclasses import dataclass
from typing import ClassVar

from cneuromax.deeplearning.common.optimizer.base import BaseOptimizerConfig


@dataclass
class AdamConfig(BaseOptimizerConfig):
    """https://pytorch.org/docs/stable/generated/torch.optim.Adam.html.

    Attributes:
        _target_: Optimizer class.
        _partial_: (Need model parameters for instantiation).
        lr: Learning rate.
        weight_decay: L2 penalty.
        betas: Exponential decay rate for the 1st & 2nd moment
            estimates.
        eps: Small constant for numerical stability.
    """

    _target_: str = "torch.optim.Adam"
    betas: ClassVar[list[float]] = [0.9, 0.999]
    eps: float = 1e-08
