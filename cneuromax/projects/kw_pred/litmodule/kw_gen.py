""":class:`KWGenerationLitModule."""

import sys
from abc import ABCMeta
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import wandb
from einops import rearrange
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


@dataclass
class KWGenerationLitModuleConfig(BaseLitModuleConfig):
    """Holds :class:`KWGenerationLitModule` config values.

    Args:
        num_val_samples_wandb: The number of samples to log to\
            :mod:`wandb`.
        cfg_scales: The classifier-free guidance scales to use\
            when sampling.
        predicting: Whether the model is predicting rather than\
            fitting.
    """

    num_val_samples_wandb: An[int, ge(1)] = 3
    cfg_scales: list[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 4.0],
    )
    predicting: bool = False


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
            self.wandb_columns = ["x", "y"] + [
                f"x_hat_{cfg_scale}" for cfg_scale in self.config.cfg_scales
            ]
            self.wandb_x_wrapper = wandb.Image
            self.val_wandb_data: list[dict[str, Any]]
        self.diffusion = create_diffusion(timestep_respacing="")  # type: ignore [no-untyped-call]
        self.ema_nnmodule = EMA(
            model=self.nnmodule,
            include_online_model=False,
        )
        # self.ema_nnmodule.ema_model = torch.compile(
        #     self.ema_nnmodule.ema_model,
        # )
        self.ema_nnmodule.ema_model.eval()  # type: ignore [attr-defined]

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
        y = data["AE"] if "AE" in data else data["AF"]
        if (
            stage == "val"
            and self.global_rank == 0
            and self.config.log_val_wandb
        ):
            self.save_val_data(x=x, y=y)
        t = torch.randint(
            low=0,
            high=self.diffusion.num_timesteps,
            size=(x.shape[0],),
            device=self.device,
        )
        model = (
            self.nnmodule if stage == "train" else self.ema_nnmodule.ema_model
        )
        loss_dict = self.diffusion.training_losses(
            model,
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
        for i in range(self.config.num_val_samples_wandb):
            index = (
                self.curr_val_epoch * self.config.num_val_samples_wandb + i
            ) % len(x)
            self.val_wandb_data.append(
                {
                    "x": to_wandb_image(x[index]),
                    "y": to_wandb_image(y[index]),
                    "y_raw": y[index],
                },
            )

    def on_after_backward(self: "KWGenerationLitModule") -> None:
        """Called after loss computation and backward pass."""
        self.ema_nnmodule.update()

    def on_validation_epoch_end(
        self: "KWGenerationLitModule",
    ) -> None:
        """Called at the end of the validation epoch."""
        if self.config.log_val_wandb and self.global_rank == 0:
            num_cfg_scales = len(self.config.cfg_scales)
            cfg_scale = (
                torch.tensor(self.config.cfg_scales)
                .repeat(
                    self.config.num_val_samples_wandb,
                )
                .to(self.device)
            )
            x_big_t = torch.randn(
                self.config.num_val_samples_wandb * num_cfg_scales,
                self.nnmodule.num_klk_corners,
                self.nnmodule.klk_seq_len,
                device=self.device,
            ).to(self.device)
            y = torch.zeros(
                (
                    self.config.num_val_samples_wandb * num_cfg_scales,
                    *self.val_wandb_data[0]["y_raw"].shape,
                ),
            )
            for i in range(self.config.num_val_samples_wandb):
                y[i * num_cfg_scales : (i + 1) * num_cfg_scales] = (
                    self.val_wandb_data[i]["y_raw"]
                )
            y = y.to(self.device)
            x_hat = (
                self.diffusion.p_sample_loop(
                    self.ema_nnmodule.ema_model.forward_with_cfg,
                    x_big_t.shape,
                    x_big_t,
                    clip_denoised=False,
                    model_kwargs={"y": y, "cfg_scale": cfg_scale},
                    progress=True,
                    device=self.device,
                )
                .squeeze()
                .cpu()
            )
            if self.config.predicting:
                torchaudio.save(
                    uri="pred.wav",
                    src=x_hat.view(1, -1),
                    sample_rate=400,
                    format="wav",
                )
                sys.exit()
            for i, val_wandb_data_i in enumerate(self.val_wandb_data):
                for j, cfg_scale_j in enumerate(self.config.cfg_scales):
                    val_wandb_data_i.update(
                        {
                            f"x_hat_{cfg_scale_j}": to_wandb_image(
                                x_hat[i * num_cfg_scales + j],
                            ),
                        },
                    )
                val_wandb_data_i.pop("y_raw")
            super().on_validation_epoch_end()


def to_wandb_image(
    data: Float[Tensor, " seq_len"] | Float[Tensor, " seq_len dim_2"],
) -> wandb.Image:
    """Converts data to a buffer.

    Args:
        data: The data to be converted.

    Returns:
        The buffer containing the data.
    """
    plt.figure()
    if data.ndim == 1:
        plt.plot(np.linspace(0, len(data) - 1, len(data)), data)
        plt.ylim(-6, 6)
    else:
        plt.imshow(data.T)
    plt.axis("off")
    canvas = plt.gca().figure.canvas  # type: ignore [union-attr]
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore [union-attr]
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return wandb.Image(image)
