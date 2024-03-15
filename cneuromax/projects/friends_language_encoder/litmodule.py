""":class:`FriendsFinetuningModel`."""

from dataclasses import dataclass
from typing import Annotated as Any

from jaxtyping import Num
from peft import get_peft_model
from peft.config import PeftConfig
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
    BaseLitModuleConfig,
)
from cneuromax.friends_language_encoder.peftmodule import PEFTLitModule
from cneuromax.utils.beartype import one_of


@dataclass
class FriendsLitModuleConfig(BaseLitModuleConfig):
    """Holds :class:`FriendsLitModule` config values.

    Args:
        layer_name: layer to unfreeze
    """


class FriendsFinetuningModel(BaseLitModule):
    """``project`` :class:`~BaseLitModule`."""

    def __init__(
        self: "FriendsFinetuningModel",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config: FriendsLitModuleConfig

        for param in self.nnmodule.parameters():
            param.requires_grad = False

        for layer_name in self.config.layer_names:
            for name, param in self.nnmodule.named_parameters():
                if layer_name in name:
                    param.requires_grad = True

    def step(
        self: "FriendsFinetuningModel",
        batch: BatchEncoding,
        stage: Any[str, one_of("train", "val", "test", "predict")],
    ) -> Num[Tensor, " ..."]:
        """Inputs a batch and returns the loss or logits.

        Args:
            batch: See :paramref:`~.BaseLitModule.x_step.batch`.
            stage: See :paramref:`~.BaseLitModule.x_step.stage`.

        Returns:
            The loss if ``stage`` is ``train``, ``val``, or ``test``,\
                otherwise the logits.
        """
        out = self.nnmodule(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out: Tensor = (
            out["loss"] if stage in ["train", "val", "test"] else out["logits"]
        )
        return out


@dataclass
class FriendsPEFTModule(PEFTLitModule):
    """`project` :class:`~FriendsPEFTModule`."""

    def __init__(
        self: "FriendsPEFTModule",
        peft_config: PeftConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(peft_config, *args, **kwargs)
        self.nnmodule = get_peft_model(self.nnmodiule, peft_config)
