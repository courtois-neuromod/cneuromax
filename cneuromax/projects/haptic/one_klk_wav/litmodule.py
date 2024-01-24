""":class:`OneKLKWavLitModule`."""
from typing import Annotated as An

import torch.nn.functional as f
from jaxtyping import Float
from torch import Tensor

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
)
from cneuromax.utils.beartype import one_of


class OneKLKWavLitModule(BaseLitModule):
    """:mod:`one_klk_wav`` :class:`.BaseLitModule`."""

    def step(
        self: "OneKLKWavLitModule",
        batch: dict[str, Float[Tensor, " data_len"]],
        _: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, " "]:
        """Next time step prediction w/ MSE loss.

        Args:
            batch: The whole ``.wav`` file data.
            _: See :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The MSE loss.
        """
        inputs: Float[Tensor, " data_len-1"] = batch["BR"][:-1]
        targets: Float[Tensor, " data_len-1"] = batch["BR"][1:]
        outputs: Float[Tensor, " data_len-1"] = self.nnmodule(inputs)
        return f.mse_loss(outputs, targets)
