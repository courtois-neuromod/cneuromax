""":class:`BaseAutoencodingLitModule`."""

from typing import Annotated as An

import torch.nn.functional as f
from jaxtyping import Float
from torch import Tensor

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.fitting.deeplearning.litmodule.nnmodule.autoencoder import (
    BaseAutoEncoder,
)
from cneuromax.utils.beartype import one_of


class BaseAutoencodingLitModule(BaseLitModule):
    """Base Autoencoding :class:`.BaseLitModule`.

    Args:
        nnmodule: See :class:`.BaseAutoEncoder`.
    """

    def __init__(
        self: "BaseAutoencodingLitModule",
        nnmodule: BaseAutoEncoder,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(nnmodule, *args, **kwargs)

    def step(
        self: "BaseAutoencodingLitModule",
        batch: Float[Tensor, " batch_size *data_dim"],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, ""]:
        """Computes the mean squared error loss for the autoencoder.

        Args:
            batch: The input data batch.
            stage: See\
                :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The mean squared error loss.
        """
        x = batch
        recon_x = self.nnmodule(x)
        recon_loss = f.mse_loss(x, recon_x)
        self.log(name=f"{stage}/mse", value=recon_loss)
        return recon_loss
