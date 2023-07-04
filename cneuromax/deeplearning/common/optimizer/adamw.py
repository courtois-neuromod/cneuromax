"""."""

from dataclasses import dataclass

from cneuromax.deeplearning.common.optimizer import AdamConfig


@dataclass
class AdamWConfig(AdamConfig):
    """https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html.

    Attributes:
        _target_: Optimizer class.
        _partial_: (Need model parameters for instantiation).
        lr: Learning rate.
        weight_decay: L2 penalty.
        betas: Exponential decay rate for the 1st & 2nd moment
            estimates.
        eps: Small constant for numerical stability.
    """

    _target_: str = "torch.optim.AdamW"
