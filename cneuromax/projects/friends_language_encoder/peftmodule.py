""":class:`PEFTLitModule`."""

from abc import ABC
from dataclasses import dataclass
from typing import Any

from peft.config import PeftConfig
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
    BaseLitModuleConfig,
)


@dataclass
class PEFTLitModule(BaseLitModule, ABC):
    """``project`` :class:`~PEFTLitModule`."""

    def __init__(
        self: "PEFTLitModule",
        config: "BaseLitModuleConfig",
        peft_config: "PeftConfig",
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> None:
        """."""
        super().__init__(
            config,
            peft_config,
            *args,
            **kwargs,
        )
        self.config = config
        self.peft_config = peft_config

        lora_config = LoraConfig(
            task_type=self.peft_config.task_type,
            lora_alpha=self.peft_config.lora_alpha,
            r=self.peft_config.lora_rank,
            lora_dropout=self.peft_config.lora_dropout,
        )
        self.nnmodule = get_peft_model(self.nnmodule, lora_config)
