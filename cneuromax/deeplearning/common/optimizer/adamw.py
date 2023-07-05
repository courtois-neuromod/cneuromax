"""."""

from dataclasses import dataclass

from hydra_zen import store

from cneuromax.deeplearning.common.optimizer.adam import AdamConfig


@store(name="adamw", group="optimizer")
@dataclass
class AdamWConfig(AdamConfig):
    """https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html.

    Attributes:
        _target_: Optimizer class.
    """

    _target_: str = "torch.optim.AdamW"
