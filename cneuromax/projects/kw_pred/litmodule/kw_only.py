""":class:`KWOnlyLitModule`."""

from typing import Annotated as An

import torch
import torch.nn.functional as f
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
)
from cneuromax.utils.beartype import one_of


class KWOnlyLitModule(BaseLitModule):
    """:mod:`.one_klk_wav` :class:`.BaseLitModule`."""

    def step(
        self: "KWOnlyLitModule",
        batch: dict[str, Float[Tensor, " batch_size num_freqs seq_len"]],
        _: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, " "]:
        """Next time step prediction w/ MSE loss.

        Args:
            batch: The whole ``.wav`` file data.
            _: See :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The MSE loss.
        """
        x: Float[Tensor, " data_len 1"] = rearrange(batch["BR"], "b n -> n b")
        inputs: Float[Tensor, " data_len_minus_1 1"] = x[:-1]
        targets: Float[Tensor, " data_len_minus_1 1"] = x[1:]
        outputs: Float[Tensor, " data_len_minus_1 1"] = torch.tanh(
            input=self.nnmodule(inputs),
        )
        return f.mse_loss(outputs, targets)
