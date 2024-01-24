""":class:`FriendsFinetuningModel`."""
from typing import Annotated as An
from typing import Any

from jaxtyping import Num
from torch import Tensor

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.beartype import one_of


class FriendsFinetuningModel(BaseLitModule):
    """``project`` :class:`~BaseLitModule`."""

    def __init__(
        self: "FriendsFinetuningModel",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)
        for name, param in self.nnmodule.named_parameters():
            if (
                "cls.predictions.transform.dense" in name
                or "vocab_transform" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def step(
        self: "FriendsFinetuningModel",
        batch: dict[str, Num[Tensor, " ..."]],
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
