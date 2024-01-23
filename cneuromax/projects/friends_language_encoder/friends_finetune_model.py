"""."""
from typing import Annotated as An

from jaxtyping import Num
from torch import Tensor

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.beartype import one_of


class FriendsFinetuningModel(BaseLitModule):
    """."""

    def freeze_layers(self: "FriendsFinetuningModel") -> None:
        """.

        Freezes all layers of the model
        """
        for param in self.nnmodule.parameters():
            param.requires_grad = False

    def unfreeze_layers(self: "FriendsFinetuningModel") -> None:
        """.

        Unfreezes last layer of the model

        Args:
            layer_idx: the index of the layer that you would like to
            freeze up until.
        """
        # Unfreeze last N transformer layers (e.g. Bert)
        for layer in self.transformer_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    def step(
        self: "FriendsFinetuningModel",
        batch: tuple[Num[Tensor, " ..."], ...],
        stage: An[str, one_of("train", "val", "test", "predict")],
    ) -> Num[Tensor, " ..."]:
        """.

        Runs the training, validation and test steps based on the
        stage.

        Args:
            batch: .
            stage: define training, validation, test or predict stage
                to return the loss
        Returns:
            output.loss or output.logit based on the stage

        """
        outputs = self.nnmodule(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
            labels=batch["labels"],
        )

        # Return value based on the stage
        if stage == "predict":
            # In prediction, return logits as there are no labels
            return outputs.logits

        # For train, val, and test stages, return the loss
        return outputs.loss
