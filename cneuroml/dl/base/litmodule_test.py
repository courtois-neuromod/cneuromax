"""Tests for the Base LitModule.

Abbreviations used in this module:

Lightning's ``LightningModule`` is short for
``lightning.pytorch.LightningModule``.

PyTorch ``nn.Module`` is short for ``torch.nn.Module``.

PyTorch ``Optimizer`` is short for ``torch.optim.Optimizer``.

PyTorch ``LRScheduler`` is short for
``torch.optim.lr_scheduler.LRScheduler``.

``Float`` is short for ``jaxtyping.Float``.
"""

from functools import partial
from typing import Literal

import pytest
import torch
from beartype import beartype as typechecker
from jaxtyping import Float
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuroml.dl.base import BaseLitModule


@pytest.fixture()
def nnmodule() -> nn.Module:
    """Creates and returns a generic PyTorch ``nn.Module`` instance.

    Returns:
        A generic PyTorch ``nn.Module`` instance.
    """
    return nn.Linear(1, 1)


@pytest.fixture()
def optimizer_partial() -> partial[Optimizer]:
    """Returns a generic PyTorch ``Optimizer`` partial function.

    Returns:
        A generic PyTorch ``Optimizer`` partial function.
    """
    return partial(torch.optim.Adam, lr=0.01)


@pytest.fixture()
def scheduler_partial() -> partial[LRScheduler]:
    """Returns a generic PyTorch ``LRScheduler`` partial function.

    Returns:
        A generic PyTorch ``LRScheduler`` partial function.
    """
    return partial(torch.optim.lr_scheduler.StepLR, step_size=1)


class GenericLitModule(BaseLitModule):
    """Generic Lightning Module.

    Attributes:
        nnmodule (``nn.Module``): The PyTorch ``nn.Module`` instance.
        optimizer (``Optimizer``): The PyTorch ``Optimizer`` instance.
        scheduler (``LRScheduler``): The PyTorch ``LRScheduler``
            instance.
    """

    @typechecker
    def step(
        self: "GenericLitModule",
        batch: Tensor | tuple[Tensor],
        stage: Literal["train", "val", "test"],
    ) -> Float[Tensor, ""]:
        """Step method common to all stages.

        Args:
            batch: An input data batch (images/sound/language/...).
            stage: The current stage (train/val/test).

        Returns:
            The loss value.
        """
        if batch:
            pass

        if stage == "train":
            loss = torch.tensor(0.0)
        elif stage == "val":
            loss = torch.tensor(1.0)
        else:  # stage == "test":
            loss = torch.tensor(2.0)

        return loss


def test_constructor(
    nnmodule: nn.Module,
    optimizer_partial: partial[Optimizer],
    scheduler_partial: partial[LRScheduler],
) -> None:
    """Tests ``__init__`` method.

    Args:
        nnmodule: A PyTorch ``nn.Module`` instance.
        optimizer_partial: A PyTorch ``Optimizer`` partial function.
        scheduler_partial: A PyTorch ``LRScheduler`` partial function.
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
    optimizer_partial: partial[Optimizer],
    scheduler_partial: partial[LRScheduler],
) -> BaseLitModule:
    """Creates and returns a generic ``BaseLitModule`` instance.

    Args:
        nnmodule: A PyTorch ``nn.Module`` instance.
        optimizer_partial: A PyTorch ``Optimizer`` partial function.
        scheduler_partial: A PyTorch ``LRScheduler`` partial function.

    Returns:
        A generic ``BaseLitModule`` instance.
    """
    return GenericLitModule(nnmodule, optimizer_partial, scheduler_partial)


@pytest.fixture()
def no_step_litmodule(
    nnmodule: nn.Module,
    optimizer_partial: partial[Optimizer],
    scheduler_partial: partial[LRScheduler],
) -> BaseLitModule:
    """Creates & returns a ``BaseLitModule`` instance with no ``step``.

    This object is used to test the ``BaseLitModule`` functionality
    when the ``step`` instance method is not implemented.

    Args:
        nnmodule: A PyTorch ``nn.Module`` instance.
        optimizer_partial: A PyTorch ``Optimizer`` partial function.
        scheduler_partial: A PyTorch ``LRScheduler`` partial function.

    Returns:
        A ``BaseLitModule`` instance with no ``step`` instance method
        implemented.
    """
    return BaseLitModule(nnmodule, optimizer_partial, scheduler_partial)


def test_training_step(litmodule: GenericLitModule) -> None:
    """Tests ``training_step`` method.

    Args:
        litmodule: A generic ``BaseLitModule`` instance.
    """
    loss = litmodule.training_step(torch.tensor(0.0))
    assert torch.isclose(loss, torch.tensor(0.0))


def test_training_step_no_step(no_step_litmodule: BaseLitModule) -> None:
    """Tests ``training_step`` method with no ``step``.

    Args:
        no_step_litmodule: A ``BaseLitModule`` instance with no ``step``
            instance method implemented.
    """
    with pytest.raises(AttributeError):
        no_step_litmodule.training_step(torch.tensor(0.0))


def test_validation_step(litmodule: GenericLitModule) -> None:
    """Tests ``validation_step`` method.

    Args:
        litmodule: A generic ``BaseLitModule`` instance.
    """
    loss = litmodule.validation_step(torch.tensor(0.0))
    assert torch.isclose(loss, torch.tensor(1.0))


def test_validation_step_no_step(no_step_litmodule: BaseLitModule) -> None:
    """Tests ``validation_step`` method with no ``step``.

    Args:
        no_step_litmodule: A ``BaseLitModule`` instance with no ``step``
            instance method implemented.
    """
    with pytest.raises(AttributeError):
        no_step_litmodule.validation_step(torch.tensor(0.0))


def test_test_step(litmodule: GenericLitModule) -> None:
    """Tests ``test_step`` method.

    Args:
        litmodule: A generic ``BaseLitModule`` instance.
    """
    loss = litmodule.test_step(torch.tensor(0.0))
    assert torch.isclose(loss, torch.tensor(2.0))


def test_test_step_no_step(no_step_litmodule: BaseLitModule) -> None:
    """Tests ``test_step`` method with no ``step``.

    Args:
        no_step_litmodule: A ``BaseLitModule`` instance with no ``step``
            instance method implemented.
    """
    with pytest.raises(AttributeError):
        no_step_litmodule.test_step(torch.tensor(0.0))


def test_configure_optimizers(litmodule: GenericLitModule) -> None:
    """Tests ``configure_optimizers`` method.

    Args:
        litmodule: A generic ``BaseLitModule`` instance.
    """
    [optimizer], [scheduler] = litmodule.configure_optimizers()
    assert optimizer == litmodule.optimizer
    assert scheduler == litmodule.scheduler
