""":class:`FriendsFinetuningModel`."""

from dataclasses import dataclass
from typing import Annotated as An
from typing import Any

from jaxtyping import Num
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
)
from cneuromax.utils.beartype import one_of


@dataclass
class FriendsLitModuleConfig:
    """Holds :class:`FriendsLitModule` config values.

    Args:
        layer_name: layer to unfreeze
    """

    layer_names: list[str, str] = "${layer_names}"
    num_layer_to_freeze: int = "${num_layer_to_freeze}"


class FriendsFinetuningModel(BaseLitModule):
    """``project`` :class:`~BaseLitModule`."""

    def __init__(
        self: "FriendsFinetuningModel",
        config: FriendsLitModuleConfig,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)

        # for _, param in self.nnmodule.named_parameters():
        #     param.requires_grad = False

        # First, freeze all parameters in the model
        for i, block in enumerate(self.nnmodule.transformer.h):
            # Only un-freeze the last n transformer blocks
            if i < config.num_layer_to_freeze:
                for parameter in block.parameters():
                    parameter.requires_grad = False

        # Then, selectively unfreeze the specified layers
        for layer_name in config.layer_names:
            # Check if the layer_name directly matches a nodel attribute
            if hasattr(self.nnmodule, layer_name):
                layer = getattr(self.nnmodule, layer_name)
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            else:
                # If the layer_name doesn't directly match, but a part
                for name, param in self.nnmodule.named_parameters():
                    if layer_name in name:
                        param.requires_grad = True

    def step(
        self: "FriendsFinetuningModel",
        batch: BatchEncoding,
        stage: An[str, one_of("train", "val", "test", "predict")],
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

    # peft_config = LoraConfig(
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    #     r=64,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
