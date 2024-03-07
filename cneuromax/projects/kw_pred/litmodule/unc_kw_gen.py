""":class:`UnconditionalKWGenerationLitModule."""

import io
from abc import ABCMeta
from typing import Annotated as An
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb
from denoising_diffusion_pytorch import GaussianDiffusion1D
from einops import rearrange
from ema_pytorch import EMA
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.beartype import one_of


class UnconditionalKWGenerationLitModule(BaseLitModule, metaclass=ABCMeta):
    """Unconditional ``.klk`` ``.wav``generation ``LitModule``.

    Attributes:
        accuracy\
            (:class:`~torchmetrics.generation.MulticlassAccuracy`)
        wandb_input_data_wrapper (:`callable`): A wrapper to be used\
            around the input datapoint when logging to W&B.
        wandb_table: A W&B table to store validation data.
    """

    def __init__(
        self: "UnconditionalKWGenerationLitModule",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.config.log_val_wandb:
            self.wandb_columns = ["x", "x_hat"]
            self.val_wandb_data: list[dict[str, Tensor]]
        self.diffusion_module = GaussianDiffusion1D(
            model=self.nnmodule,
            seq_length=800,
            sampling_timesteps=100,
        )
        self.ema = EMA(model=self.nnmodule)

    def step(
        self: "UnconditionalKWGenerationLitModule",
        data: dict[str, Tensor],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, " "]:
        """Computes the model accuracy and cross entropy loss.

        Args:
            data: .
            stage: See\
                :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The cross entropy loss.
        """
        x: Float[Tensor, " batch_size seq_len"] = data["KW BL"]
        x: Float[Tensor, " batch_size 1 seq_len"] = rearrange(
            tensor=x,
            pattern="BS SL -> BS 1 SL",
        )
        if stage == "val" and self.config.log_val_wandb:
            self.save_val_data(x=x)
        loss: Float[Tensor, ""] = self.diffusion_module.forward(img=x)
        return loss

    def save_val_data(
        self: "UnconditionalKWGenerationLitModule",
        x: Float[Tensor, " batch_size 1 seq_len"],
    ) -> None:
        """Saves data computed during validation for later use.

        Args:
            x: The input data.
            y: The target data.
            logits: The network's raw output.
            preds: The model's predictions.
        """
        x: Float[Tensor, " batch_size seq_len"] = x.squeeze().cpu()
        for x_i in x:
            self.val_wandb_data.append({"x": x_i})

    def on_after_backward(self: "UnconditionalKWGenerationLitModule") -> None:
        """Called after loss computation and backward pass."""
        self.ema.update()

    def on_validation_epoch_end(
        self: "UnconditionalKWGenerationLitModule",
    ) -> None:
        """Called at the end of the validation epoch."""
        self.diffusion_module.model = self.ema.ema_model
        x_hat: Float[Tensor, " batch_size seq_len"] = (
            self.diffusion_module.sample(batch_size=3).squeeze()
        )
        self.diffusion_module.model = self.nnmodule
        self.val_wandb_data = self.val_wandb_data[
            (self.curr_val_epoch * 3)
            % len(self.val_wandb_data) : ((self.curr_val_epoch + 1) * 3)
            % len(self.val_wandb_data)
        ]
        for val_wandb_data_i, x_hat_i in zip(
            self.val_wandb_data,
            x_hat,
            strict=True,
        ):
            val_wandb_data_i.update({"x_hat": x_hat_i})
        super().on_validation_epoch_end()


def to_wandb_image(data: Float[Tensor, " seq_len"]) -> wandb.Image:
    """Converts data to a buffer.

    Args:
        data: The data to be converted.

    Returns:
        The buffer containing the data.
    """
    plt.figure()
    plt.plot(np.linspace(0, len(data) - 1, len(data)), data)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    buf.close()
    return wandb.Image(im)
