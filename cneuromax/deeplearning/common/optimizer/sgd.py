"""."""

from dataclasses import dataclass

from hydra_zen import store

from cneuromax.deeplearning.common.optimizer.base import BaseOptimizerConfig


@store(name="sgd", group="optimizer")
@dataclass
class AdamConfig(BaseOptimizerConfig):
    """https://pytorch.org/docs/stable/generated/torch.optim.SGD.html.

    Attributes:
        _target_: Optimizer class.
        momentum: .
    """

    _target_: str = "torch.optim.SGD"
    momentum: float = 0
