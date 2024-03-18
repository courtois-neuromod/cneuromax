""":class:`PEFTLitModule`."""

from abc import ABC
from dataclasses import dataclass
from functools import partial

from peft import LoraConfig, get_peft_model
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
    BaseLitModuleConfig,
)


@dataclass
class PEFTLitModule(BaseLitModule, ABC):
    """``project`` :class:`~PEFTLitModule`."""

    def __init__(
        self: "PEFTLitModule",
        config: BaseLitModuleConfig,
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        """."""
        super().__init__(config, nnmodule, optimizer, scheduler)
        self.config = config
        peft_config = LoraConfig(
            task_type=self.config.task_type,
            lora_alpha=self.config.lora_alpha,
            r=self.config.local_rank,
            inference_mode=self.config.inference_mode,
            lora_dropout=self.config.lora_dropout,
        )
        self.nnmodule = get_peft_model(nnmodule, peft_config)
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
