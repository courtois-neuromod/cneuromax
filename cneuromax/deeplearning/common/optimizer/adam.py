"""."""

from dataclasses import dataclass

from hydra_zen import store

from cneuromax.deeplearning.common.optimizer.base import BaseOptimizerConfig


@store(name="adam", group="optimizer")
@dataclass
class AdamConfig(BaseOptimizerConfig):
    """https://pytorch.org/docs/stable/generated/torch.optim.Adam.html.

    Attributes:
        _target_: Optimizer class.
        betas: Exponential decay rate for the 1st & 2nd moment
            estimates.
        eps: Small constant for numerical stability.
    """

    _target_: str = "torch.optim.Adam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
