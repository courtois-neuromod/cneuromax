""":class:`KWGenerationLitModule."""

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
from cneuromax.utils.beartype import ge, one_of

from .dit import CustomDiT
from .unc_kw_gen import to_wandb_image


@dataclass
class KWGenerationLitModuleConfig(BaseLitModuleConfig):
    """Holds :class:`KWGenerationLitModule` config values.

    Args:
        num_val_wandb_samples: The number of samples to log to\
            :mod:`wandb`.
        randomize_conditioning: Whether to replace the conditioning\
            information with random values.
    """

    num_val_wandb_samples: An[int, ge(1)] = 3
    conditioning: An[str, one_of("random", "linear", "transformer")] = "random"


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
            self.val_wandb_data: list[dict[str, Any]]
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
        if self.config.conditioning == "random":
            y: Float[Tensor, " BS AES"] = torch.randn(
                (x.shape[0], self.nnmodule.hidden_size),
                device=self.device,
            )
        else:
            y = data["AE"]
            if (
                self.config.conditioning == "linear"
                and y.dim() == 3  # noqa: PLR2004
            ):
                y: Float[Tensor, " BS AES"] = reduce(  # type: ignore [no-redef]
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
        y: Tensor,  # Float[Tensor, " batch_size mean_num_ae"],
    ) -> None:
        """Saves data computed during validation for later use.

        Args:
            x: The input data.
            y: The target data.
            logits: The network's raw output.
            preds: The model's predictions.
        """
        if self.val_wandb_data:
            return
        x: Float[Tensor, " batch_size seq_len"] = x.squeeze().cpu()
        y: Tensor = y.squeeze().cpu()
        for i, (x_i, y_i) in enumerate(zip(x, y, strict=True)):
            self.val_wandb_data.append(
                {"x": to_wandb_image(x_i), "y": y_i},
            )
            if i + 1 == self.config.num_val_wandb_samples:
                break

    def on_after_backward(self: "KWGenerationLitModule") -> None:
        """Called after loss computation and backward pass."""
        self.ema.update()

    def on_validation_epoch_end(
        self: "KWGenerationLitModule",
    ) -> None:
        """Called at the end of the validation epoch."""
        if self.config.log_val_wandb:
            x_big_t = torch.randn(
                self.config.num_val_wandb_samples,
                self.nnmodule.in_channels,
                self.nnmodule.input_size,
                device=self.device,
            )
            y = torch.empty(
                (
                    self.config.num_val_wandb_samples,
                    *self.val_wandb_data[0]["y"].shape,
                ),
            )
            for i in range(self.config.num_val_wandb_samples):
                y[i] = self.val_wandb_data[i]["y"]
            y = y.to(self.device)
            x_zero_hat = self.diffusion.p_sample_loop(
                self.nnmodule.forward,
                x_big_t.shape,
                x_big_t,
                clip_denoised=False,
                model_kwargs={"y": y},
                progress=True,
                device=self.device,
            )
            x_zero_hat = x_zero_hat.squeeze().cpu()
            for val_wandb_data_i, x_hat_i in zip(
                self.val_wandb_data,
                x_zero_hat,
                strict=True,
            ):
                val_wandb_data_i.update({"x_hat": to_wandb_image(x_hat_i)})
                val_wandb_data_i.pop("y")
        super().on_validation_epoch_end()
