""":class:`FriendsFinetuningModel`."""

from dataclasses import dataclass, field
from typing import Any, list

from jaxtyping import Num
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModuleConfig,
)
from cneuromax.projects.friends_language_encoder.peftmodule import (
    PEFTLitModule,
)


@dataclass
class FriendsLitModuleConfig(BaseLitModuleConfig):
    """Holds :class:`FriendsLitModule` config values.

    Args:
        layer_name: layer to unfreeze
    """

    layer_names: list[str] = field(default_factory=list)


class FriendsFinetuningModel(PEFTLitModule):
    """``project`` :class:`~BaseLitModule`."""

    def __init__(
        self: "FriendsFinetuningModel",
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config: FriendsLitModuleConfig

    def step(
        self: "FriendsFinetuningModel",
        batch: BatchEncoding,
    ) -> Num[Tensor, " ..."]:
        """Inputs a batch and returns the loss or logits.

        Args:
            batch: See :paramref:`~.BaseLitModule.x_step.batch`.
            stage: See :paramref:`~.BaseLitModule.x_step.stage`.

        Returns:
            Returns the loss for either stage [train`, val, test].
        """
        out = self.nnmodule(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return out.loss

    def predict_step(
        self: "FriendsFinetuningModel",
        batch: BatchEncoding,
    ) -> Num[Tensor, " ..."]:
        """Processes a batch and returns the logits for prediction.

        Args:
            batch: The input batch.

        Returns:
            The logits as a tensor.
        """
        out = self.nnmodule(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return out.logits
