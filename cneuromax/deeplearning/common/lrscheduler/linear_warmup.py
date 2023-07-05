"""."""

from dataclasses import dataclass

from hydra_zen import store

from cneuromax.deeplearning.common.lrscheduler.base import (
    BaseLRSchedulerConfig,
)


@store(name="linear_warmup", group="lrscheduler")
@dataclass
class LinearWarmupConfig(BaseLRSchedulerConfig):
    """https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_constant_schedule_with_warmup.

    Attributes:
        _target_: .
        num_warmup_steps: .
    """

    _target_: str = "transformers.get_constant_schedule_with_warmup"
    num_warmup_steps: int = 0
