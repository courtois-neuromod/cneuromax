""":class:`KWGenerationLitModule."""

from abc import ABCMeta
from typing import Annotated as An
from typing import Any

import torch
import wandb
from einops import rearrange
from ema_pytorch import EMA
from jaxtyping import Float
from torch import Tensor

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.beartype import one_of

from ..dit.diffusion import create_diffusion  # noqa: TID252
from .unc_kw_gen import to_wandb_image


class KWGenerationLitModule(BaseLitModule, metaclass=ABCMeta):
    """``.klk`` ``.wav``generation ``LitModule``."""

    def __init__(
        self: "KWGenerationLitModule",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)
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

        Args:
            data: .
            stage: See\
                :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The cross entropy loss.
        """
        x: Float[Tensor, " batch_size 1 4000"] = rearrange(
            tensor=data["KW BL"],
            pattern="BS SL -> BS 1 SL",
        )
        if stage == "val" and self.config.log_val_wandb:
            self.save_val_data(x=x)
        t = torch.randint(
            0,
            self.diffusion.num_timesteps,
            (x.shape[0],),
            device=self.device,
        )
        loss_dict = self.diffusion.training_losses(
            self.nnmodule,
            x,
            t,
            {"y": data["AF"]},
        )
        loss: Float[Tensor, ""] = loss_dict["loss"].mean()
        return loss

    def save_val_data(
        self: "KWGenerationLitModule",
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
            self.val_wandb_data.append({"x": to_wandb_image(x_i)})

    def on_after_backward(self: "KWGenerationLitModule") -> None:
        """Called after loss computation and backward pass."""
        self.ema.update()

    def on_validation_epoch_end(
        self: "KWGenerationLitModule",
    ) -> None:
        """Called at the end of the validation epoch."""
        if self.config.log_val_wandb:
            """
            # Labels to condition the model with (feel free to change):
            class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

            # Create sampling noise:
            n = len(class_labels)
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = torch.tensor(class_labels, device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            """
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
                val_wandb_data_i.update({"x_hat": to_wandb_image(x_hat_i)})
        super().on_validation_epoch_end()
