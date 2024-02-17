""":class:`MNISTGenerationLitModule` & its config."""

from abc import ABCMeta
from collections.abc import Callable  # noqa: TCH003
from copy import copy
from functools import partial
from typing import Annotated as An
from typing import Any

import wandb
from denoising_diffusion_pytorch import GaussianDiffusion
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.beartype import one_of


class MNISTGenerationLitModule(BaseLitModule, metaclass=ABCMeta):
    """MNIST Generation :mod:`lightning` ``LitModule``.

    Args:
        config: See :class:`BaseGenerationLitModuleConfig`.
        nnmodule: See :paramref:`~.BaseLitModule.nnmodule`.
        optimizer: See :paramref:`~.BaseLitModule.optimizer`.
        scheduler: See :paramref:`~.BaseLitModule.scheduler`.

    Attributes:
        accuracy\
            (:class:`~torchmetrics.generation.MulticlassAccuracy`)
        wandb_input_data_wrapper (:`callable`): A wrapper to be used\
            around the input datapoint when logging to W&B.
        wandb_table: A W&B table to store validation data.
    """

    def __init__(
        self: "MNISTGenerationLitModule",
        nnmodule: nn.Module,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
    ) -> None:
        super().__init__(
            nnmodule=nnmodule,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        # W&B validation attributes.
        self.wandb_input_data_wrapper = wandb.Image
        self.wandb_table: wandb.Table = wandb.Table(  # type: ignore[no-untyped-call]
            columns=["idx", "epoch", "x", "preds"],
        )
        self.diffusion_module = GaussianDiffusion(
            model=self.nnmodule,
            image_size=28,
        )

    def step(
        self: "MNISTGenerationLitModule",
        batch: tuple[
            Float[Tensor, " batch_size *x_shape"],
            Int[Tensor, " batch_size"],
        ],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, " "]:
        """Computes the model accuracy and cross entropy loss.

        Args:
            batch: A tuple ``(X, y)`` where ``X`` is the input data and\
                ``y`` is the target data.
            stage: See\
                :paramref:`~.BaseLitModule.stage_step.stage`.

        Returns:
            The cross entropy loss.
        """
        x: Float[Tensor, " batch_size *x_shape"] = batch[0]
        if stage == "val":
            self.save_val_data(x=x)
        loss: Tensor = self.diffusion_module.forward(img=x)
        return loss

    def save_val_data(
        self: "MNISTGenerationLitModule",
        x: Float[Tensor, " batch_size *x_shape"],
    ) -> None:
        """Saves data computed during validation for later use.

        Args:
            x: The input data.
            y: The target data.
            logits: The network's raw output.
            preds: The model's predictions.
        """

        x = x.cpu().numpy()
        for x_i in x:
            self.val_data.append(x_i)

    def on_validation_epoch_end(self: "MNISTGenerationLitModule") -> None:
        """Called at the end of the validation epoch."""

        for i, x_i in enumerate(self.val_data):
            pred_i = self.diffusion_module.sample(batch_size=1)
            self.wandb_table.add_data(  # type: ignore[no-untyped-call]
                i,
                self.curr_val_epoch,
                self.wandb_input_data_wrapper(x_i),
                self.wandb_input_data_wrapper(pred_i),
            )
            if i >= 2:
                break
        # 1) Static type checking discrepancy:
        # `logger.experiment` is a `wandb.wandb_run.Run` instance.
        # 2) Cannot log the same table twice:
        # https://github.com/wandb/wandb/issues/2981#issuecomment-1458447291
        self.logger.experiment.log({"val_data": copy(self.wandb_table)})  # type: ignore[union-attr]
        super().on_validation_epoch_end()
