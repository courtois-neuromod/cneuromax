"""."""

from functools import partial
from typing import Literal

import pytest
import torch
from beartype import beartype as typechecker
from jaxtyping import Float
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cneuroml.deeplearning.common.litmodule import BaseLitModule


@pytest.fixture()
def nnmodule() -> nn.Module:
    """.

    Returns:
        A generic PyTorch ``nn.Module`` instance.
    """
    return nn.Linear(5, 3)


@pytest.fixture()
def optimizer_partial() -> partial[Optimizer]:
    """.

    Returns:
        A generic PyTorch ``Optimizer`` partial function.
    """
    return partial(torch.optim.Adam, lr=0.01)


@pytest.fixture()
def scheduler_partial() -> partial[LRScheduler]:
    """.

    Returns:
        A generic PyTorch ``LRScheduler`` partial function.
    """
    return partial(torch.optim.lr_scheduler.StepLR, step_size=1)


class GenericLitModule(BaseLitModule):
    """.

    Attributes:
        nnmodule (``nn.Module``): .
        optimizer (``Optimizer``): .
        scheduler (``LRScheduler``): .
    """

    @typechecker
    def step(
        self: "GenericLitModule",
        batch: Tensor | tuple[Tensor],
        stage: Literal["train", "val", "test"],
    ) -> Float[Tensor, " "]:
        """Step method common to all stages.

        Args:
            batch: .
            stage: .

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
    """.

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
    """.

    Args:
        nnmodule: A PyTorch ``nn.Module`` instance.
        optimizer_partial: A PyTorch ``Optimizer`` partial function.
        scheduler_partial: A PyTorch ``LRScheduler`` partial function.

    Returns:
        A generic ``LitModule`` instance.
    """
    return GenericLitModule(nnmodule, optimizer_partial, scheduler_partial)


@pytest.fixture()
def litmodule_with_no_step_method(
    nnmodule: nn.Module,
    optimizer_partial: partial[Optimizer],
    scheduler_partial: partial[LRScheduler],
) -> BaseLitModule:
    """.

    This object is used to test the ``LitModule`` functionality
    when the ``step`` instance method is not implemented.

    Args:
        nnmodule: A PyTorch ``nn.Module`` instance.
        optimizer_partial: A PyTorch ``Optimizer`` partial function.
        scheduler_partial: A PyTorch ``LRScheduler`` partial function.

    Returns:
        A ``LitModule`` instance with no ``step`` instance method
        implemented.
    """
    return BaseLitModule(nnmodule, optimizer_partial, scheduler_partial)


def test_training_step(litmodule: GenericLitModule) -> None:
    """.

    Args:
        litmodule: .
    """
    loss = litmodule.training_step(torch.tensor(0.0))
    assert torch.isclose(loss, torch.tensor(0.0))


def test_training_step_no_step_method(
    litmodule_with_no_step_method: BaseLitModule,
) -> None:
    """.

    Args:
        litmodule_with_no_step_method: .
    """
    with pytest.raises(AttributeError):
        litmodule_with_no_step_method.training_step(torch.tensor(0.0))


def test_validation_step(litmodule: GenericLitModule) -> None:
    """.

    Args:
        litmodule: .
    """
    loss = litmodule.validation_step(torch.tensor(0.0))
    assert torch.isclose(loss, torch.tensor(1.0))


def test_validation_step_no_step_method(
    litmodule_with_no_step_method: BaseLitModule,
) -> None:
    """.

    Args:
        litmodule_with_no_step_method: .
    """
    with pytest.raises(AttributeError):
        litmodule_with_no_step_method.validation_step(torch.tensor(0.0))


def test_test_step(litmodule: GenericLitModule) -> None:
    """.

    Args:
        litmodule: .
    """
    loss = litmodule.test_step(torch.tensor(0.0))
    assert torch.isclose(loss, torch.tensor(2.0))


def test_test_step_no_step_method(
    litmodule_with_no_step_method: BaseLitModule,
) -> None:
    """.

    Args:
        litmodule_with_no_step_method: .
    """
    with pytest.raises(AttributeError):
        litmodule_with_no_step_method.test_step(torch.tensor(0.0))


def test_configure_optimizers(litmodule: GenericLitModule) -> None:
    """.

    Args:
        litmodule: .
    """
    [optimizer], [scheduler] = litmodule.configure_optimizers()
    assert optimizer == litmodule.optimizer
    assert scheduler == litmodule.scheduler
