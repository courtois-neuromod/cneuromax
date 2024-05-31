""":class:`FriendsFinetuningModel`."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple

import torch
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
        # for name, param in self.nnmodule.named_parameters():
        #     print(name)

    def step(
        self: "FriendsFinetuningModel",
        batch: BatchEncoding,
        stage: Literal["train", "val", "test"],
    ) -> Tuple[torch.Tensor, Optional[Any]]:
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
        hidden_states = out.hidden_states
        print(f"hidden_states = {hidden_states}")


        loss: Tensor = out.loss
        return loss, hidden_states

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

    # def forward(self: "FriendsFinetuningModel",
    #             input_ids: torch.Tensor,
    #             )-> Num[Tensor, " ..."]:
    #     """."""

    #     return self.nnmodule(input_ids,
    #                     output_hidden_states=True)

