""":class:`KWGenerationLitModule."""

import logging
from abc import ABCMeta
from dataclasses import dataclass
from typing import Annotated as An
from typing import Any

import torch
import wandb
from einops import rearrange, reduce
from ema_pytorch import EMA
from jaxtyping import Float
from torch import Tensor

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
    BaseLitModuleConfig,
)
from cneuromax.projects.kw_pred.dit.diffusion import (
    create_diffusion,
)
from cneuromax.utils.beartype import one_of

from .dit import CustomDiT
from .unc_kw_gen import to_wandb_image


@dataclass
class KWGenerationLitModuleConfig(BaseLitModuleConfig):
    """Holds :class:`KWGenerationLitModule` config values.

    Args:
        num_val_wandb_samples: The number of samples to log to\
            :mod:`wandb`.
    """

    num_val_wandb_samples: int = 3


class KWGenerationLitModule(BaseLitModule, metaclass=ABCMeta):
    """``.klk`` ``.wav``generation ``LitModule``."""

    def __init__(
        self: "KWGenerationLitModule",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config: KWGenerationLitModuleConfig
        self.nnmodule: CustomDiT
        if self.config.log_val_wandb:
            self.wandb_columns = ["x", "x_hat"]
            self.wandb_x_wrapper = wandb.Image
            self.val_wandb_data: list[dict[str, wandb.Image]]
        self.diffusion = create_diffusion(timestep_respacing="")  # type: ignore [no-untyped-call]
        self.ema = EMA(model=self.nnmodule)

    def step(
        self: "KWGenerationLitModule",
        data: dict[str, Tensor],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, " "]:
        """Computes the model accuracy and cross entropy loss.

        KW: ``.klk`` ``.wav``
        BL: Back left
        BS: Batch size
        SL: Sequence length
        NAE: Number of audio embeddings (time dimension)
        AES: Audio embeddings size

        Args:
            data: .
            stage: See\
                :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The cross entropy loss.
        """
        x: Float[Tensor, " BS 1 4000"] = rearrange(
            tensor=data["KW BL"],
            pattern="BS SL -> BS 1 SL",
        )
        y: Float[Tensor, " BS NAE AES"] = data["AE"]
        y: Float[Tensor, " BS AES"] = reduce(
            tensor=y,
            pattern="BS NAE AES -> BS AES",
            reduction="mean",
        )
        if stage == "val" and self.config.log_val_wandb:
            self.save_val_data(x=x, y=y)
        t = torch.randint(
            low=0,
            high=self.diffusion.num_timesteps,
            size=(x.shape[0],),
            device=self.device,
        )
        logging.info("hi")
        loss_dict = self.diffusion.training_losses(
            self.nnmodule,
            x,
            t,
            {"y": y},
        )
        loss: Float[Tensor, ""] = loss_dict["loss"].mean()
        return loss

    def save_val_data(
        self: "KWGenerationLitModule",
        x: Float[Tensor, " batch_size 1 seq_len"],
        y: Float[Tensor, " batch_size mean_num_ae"],
    ) -> None:
        """Saves data computed during validation for later use.

        Args:
            x: The input data.
            y: The target data.
            logits: The network's raw output.
            preds: The model's predictions.
        """
        x: Float[Tensor, " batch_size seq_len"] = x.squeeze().cpu()
        y: Float[Tensor, " batch_size mean_num_ae"] = y.squeeze().cpu()
        for x_i, y_i in zip(x, y, strict=True):
            self.val_wandb_data.append(
                {"x": to_wandb_image(x_i), "y": y_i},
            )

    def on_after_backward(self: "KWGenerationLitModule") -> None:
        """Called after loss computation and backward pass."""
        self.ema.update()

    def on_validation_epoch_end(
        self: "KWGenerationLitModule",
    ) -> None:
        """Called at the end of the validation epoch."""
        if self.config.log_val_wandb:
            # Labels to condition the model with (feel free to change):
            self.val_wandb_data = self.val_wandb_data[
                (self.curr_val_epoch * self.config.num_val_wandb_samples)
                % len(self.val_wandb_data) : (
                    (self.curr_val_epoch + 1)
                    * self.config.num_val_wandb_samples
                )
                % len(self.val_wandb_data)
            ]
            x_big_t = torch.randn(
                3,
                self.nnmodule.in_channels,
                self.nnmodule.input_size,
                device=self.device,
            )
            y = torch.tensor(
                (
                    self.val_wandb_data[i]["y"]
                    for i in range(self.config.num_val_wandb_samples)
                ),
                device=self.device,
            )
            x_zero_hat = self.diffusion.p_sample_loop(
                self.nnmodule.forward,
                x_big_t.shape,
                x_big_t,
                clip_denoised=False,
                model_kwargs={"y": y},
                progress=True,
                device=self.device,
            )
            for val_wandb_data_i, x_hat_i in zip(
                self.val_wandb_data,
                x_zero_hat,
                strict=True,
            ):
                val_wandb_data_i.update({"x_hat": to_wandb_image(x_hat_i)})
                val_wandb_data_i.pop("y")
        super().on_validation_epoch_end()
