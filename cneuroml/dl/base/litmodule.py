"""Base Lightning Module."""

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Literal, final

import lightning.pytorch as pl
import torch
from beartype import beartype as typechecker
from jaxtyping import Float
from torch import nn


class BaseLitModule(pl.LightningModule, metaclass=ABCMeta):
    """Base LitModule.

    Interfaces with the abstract pl.LightningModule methods.
    Subclasses should implement the 'step' method.

    Attributes:
        nnmodule (nn.Module): The PyTorch network module.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The PyTorch
            scheduler.
    """

    def __init__(
        self: "BaseLitModule",
        nnmodule: nn.Module,
        optimizer_partial: partial[torch.optim.Optimizer],
        scheduler_partial: partial[torch.optim.lr_scheduler.LRScheduler],
    ) -> None:
        """Constructor, calls parent constructor and handles arguments.

        Args:
            nnmodule: The PyTorch network module.
            optimizer_partial: The partial PyTorch optimizer.
            scheduler_partial: The partial PyTorch scheduler.
        """
        super().__init__()

        self.nnmodule = nnmodule
        self.optimizer = optimizer_partial(params=self.parameters())
        self.scheduler = scheduler_partial(optimizer=self.optimizer)

    @abstractmethod
    @typechecker
    def step(
        self: "BaseLitModule",
        batch: torch.Tensor | tuple[torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> Float[torch.Tensor, ""]:
        """Step method common to all stages.

        Args:
            batch: Input data batch (images/sound/language/...).
            stage: Current stage (train/val/test).

        Returns:
            The loss.
        """

    @final
    @typechecker
    def training_step(
        self: "BaseLitModule",
        batch: torch.Tensor | tuple[torch.Tensor],
    ) -> Float[torch.Tensor, ""]:
        """Training step method.

        Args:
            batch: Input data batch (images/sound/language/...).

        Returns:
            The loss.
        """
        loss = self.step(batch, "train")
        self.log("train/loss", loss)

        return loss

    @final
    @typechecker
    def validation_step(
        self: "BaseLitModule",
        batch: torch.Tensor | tuple[torch.Tensor],
    ) -> Float[torch.Tensor, ""]:
        """Validation step method.

        Args:
            batch: Input data batch (images/sound/language/...).

        Returns:
            The loss.
        """
        loss = self.step(batch, "val")
        self.log("val/loss", loss)

        return loss

    @final
    @typechecker
    def test_step(
        self: "BaseLitModule",
        batch: torch.Tensor | tuple[torch.Tensor],
    ) -> Float[torch.Tensor, ""]:
        """Test step method.

        Args:
            batch: Input data batch (images/sound/language/...).

        Returns:
            The loss.
        """
        loss = self.step(batch, "test")
        self.log("test/loss", loss)

        return loss

    @final
    def configure_optimizers(
        self: "BaseLitModule",
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler.LRScheduler],
    ]:
        """Configure the scheduler and optimizer.

        Returns:
            The optimizer and scheduler objects.
        """
        return [self.optimizer], [self.scheduler]
