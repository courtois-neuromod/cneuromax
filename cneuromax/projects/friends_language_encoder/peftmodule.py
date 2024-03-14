"""``project`lora adapter config and peft model."""

from dataclasses import dataclass

from jaxtyping import Num
from peft import LoraConfig, get_peft_model
from torch import Tensor

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
    BaseLitModuleConfig,
)


@dataclass
class BasePEFTLitModuleConfig(BaseLitModuleConfig):
    """Holds :class:`BasePEFTLitModule` config values.

    Args:
        task_type: training task
        inference_mode: identifies the use of the model for inference
        lora_rank: inner dimension of the LORA matrices
        lora_alpha: a scaling factor for the weight matrix magnitude
        lora_dropout: probability of dropping parameters from the LoRA
        matrices.
    """

    task_type: str = "${task_type}"
    inference_mode: str = "${inference_mode}"
    lora_rank: int = "${rank}"
    lora_alpha: int = "${lora_alpha}"
    lora_dropout: int = "${lora_dropout}"


@dataclass
class BasePEFTLitModule(BaseLitModule):
    """``project`` :class:`~BasePEFTLitModule`."""

    def __init__(
        self: "BasePEFTLitModule",
        config: BasePEFTLitModuleConfig,
    ) -> None:
        super().__init__()
        self.config = config

    def step(
        self: "BasePEFTLitModule",
    ) -> Num[Tensor, " ..."]:
        """."""
        peft_config = LoraConfig(
            task_type=self.config.task_type,
            inference_mode=self.config.inference_mode,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
        )

        return get_peft_model(self.nnmodule, peft_config)
