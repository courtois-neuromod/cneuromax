""":class:`FriendsFinetuningModel`."""

from dataclasses import dataclass, field
from typing import Any, Literal

from jaxtyping import Num
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding

from cneuromax.projects.friends_language_encoder.peftmodule import (
    PEFTLitModule,
)


class FriendsFinetuningModel(PEFTLitModule):
    """``project`` :class:`~BaseLitModule`."""

    def __init__(
        self: "FriendsFinetuningModel",
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> None:
        super().__init__(*args, **kwargs)

    def step(
        self: "FriendsFinetuningModel",
        batch: BatchEncoding,
        stage: Literal["train", "val", "test"],
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

        loss: Tensor = out.loss
        return loss

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

        logits: Tensor = out.logits

        return logits
