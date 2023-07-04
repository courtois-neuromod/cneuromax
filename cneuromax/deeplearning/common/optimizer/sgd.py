"""."""

from dataclasses import dataclass

from cneuromax.deeplearning.common.optimizer.base import BaseOptimizerConfig


@dataclass
class SGDConfig(BaseOptimizerConfig):
    """https://pytorch.org/docs/stable/generated/torch.optim.SGD.html.

    Attributes:
        _target_: Optimizer class.
        _partial_: (Need model parameters for instantiation).
        lr: Learning rate.
        weight_decay: L2 penalty.
        momentum: .
    """

    _target_: str = "torch.optim.SGD"
    momentum: float = 0
