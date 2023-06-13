"""Tests for the Base Lightning Module."""

from functools import partial
from typing import Literal

import pytest
import torch
from beartype import beartype as typechecker
from jaxtyping import Float
from torch import nn

from cneuroml.dl.base import BaseLitModule


@pytest.fixture()
def nnmodule() -> nn.Module:
    """Instantiates and returns a generic PyTorch network module.

    Returns:
        A generic PyTorch network module.
    """
    return nn.Linear(1, 1)


@pytest.fixture()
def optimizer_partial() -> partial[torch.optim.Optimizer]:
    """Returns a generic PyTorch optimizer.

    Returns:
        A generic PyTorch optimizer partial.
    """
    return partial(torch.optim.Adam, lr=0.01)


@pytest.fixture()
def scheduler_partial() -> partial[torch.optim.lr_scheduler.LRScheduler]:
    """Returns a generic PyTorch scheduler.

    Returns:
        A generic PyTorch scheduler partial.
    """
    return partial(torch.optim.lr_scheduler.StepLR, step_size=1)


class GenericLitModule(BaseLitModule):
    """Generic Lightning Module.

    Attributes:
    nnmodule (nn.Module): The PyTorch network module.
    optimizer (torch.optim.Optimizer): The PyTorch optimizer.
    scheduler (torch.optim.lr_scheduler.LRScheduler): The PyTorch
        scheduler.
    """

    @typechecker
    def step(
        self: "GenericLitModule",
        batch: torch.Tensor | tuple[torch.Tensor],  # noqa: ARG002
        stage: Literal["train", "val", "test"],
    ) -> Float[torch.Tensor, ""]:
        """Step method common to all stages.

        Args:
            batch: Input data batch (images/sound/language/...).
            stage: Current stage (train/val/test).

        Returns:
            The loss.
        """
        if stage == "train":
            loss = torch.tensor(0.0)
        elif stage == "val":
            loss = torch.tensor(1.0)
        else:  # stage == "test":
            loss = torch.tensor(2.0)

        return loss


def test_constructor(
    nnmodule: nn.Module,
    optimizer_partial: partial[torch.optim.Optimizer],
    scheduler_partial: partial[torch.optim.lr_scheduler.LRScheduler],
) -> None:
    """Test constructor.

    Args:
        nnmodule: A generic PyTorch network module.
        optimizer_partial: A generic PyTorch optimizer partial.
        scheduler_partial: A generic PyTorch scheduler partial.
    """
    torch.manual_seed(0)

    litmodule = GenericLitModule(
        nnmodule,
        optimizer_partial,
        scheduler_partial,
    )

    assert litmodule.nnmodule == nnmodule

    torch.manual_seed(0)

    test_optimizer = optimizer_partial(params=nnmodule.parameters())
    test_scheduler = scheduler_partial(optimizer=test_optimizer)

    assert str(litmodule.optimizer.param_groups) == str(
        test_optimizer.param_groups,
    )
    assert str(litmodule.optimizer.state_dict()) == str(
        test_optimizer.state_dict(),
    )
    assert str(litmodule.scheduler.state_dict()) == str(
        test_scheduler.state_dict(),
    )


@pytest.fixture()
def litmodule(
    nnmodule: nn.Module,
    optimizer_partial: partial[torch.optim.Optimizer],
    scheduler_partial: partial[torch.optim.lr_scheduler.LRScheduler],
) -> BaseLitModule:
    """Instantiates and returns a generic Lightning Module.

    Args:
        nnmodule: A generic PyTorch network module.
        optimizer_partial: A generic PyTorch optimizer partial.
        scheduler_partial: A generic PyTorch scheduler partial.

    Returns:
        A generic Lightning Module.
    """
    return GenericLitModule(nnmodule, optimizer_partial, scheduler_partial)


def test_training_step(litmodule: GenericLitModule) -> None:
    """Test training_step.

    Args:
        litmodule: A generic Lightning Module.
    """
    loss = litmodule.training_step(torch.tensor(0.0))
    assert torch.isclose(loss, torch.tensor(0.0))


def test_validation_step(litmodule: GenericLitModule) -> None:
    """Test validation_step.

    Args:
        litmodule: A generic litmodule.
    """
    loss = litmodule.validation_step(torch.tensor(0.0))
    assert torch.isclose(loss, torch.tensor(1.0))


def test_test_step(litmodule: GenericLitModule) -> None:
    """Test test_step.

    Args:
        litmodule: A generic litmodule.
    """
    loss = litmodule.test_step(torch.tensor(0.0))
    assert torch.isclose(loss, torch.tensor(2.0))


def test_configure_optimizers(litmodule: GenericLitModule) -> None:
    """Test configure_optimizers.

    Args:
        litmodule: A generic litmodule.
    """
    [optimizer], [scheduler] = litmodule.configure_optimizers()
    assert optimizer == litmodule.optimizer
    assert scheduler == litmodule.scheduler
