"""."""

from functools import partial

import pytest
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.classification.stat_scores import MulticlassStatScores

from cneuroml.deeplearning.common.litmodule.classification import (
    BaseClasssificationLitModule,
)


@pytest.mark.usefixtures("nnmodule", "optimizer_partial", "scheduler_partial")
def test_constructor(
    nnmodule: nn.Module,
    optimizer_partial: partial[Optimizer],
    scheduler_partial: partial[LRScheduler],
) -> None:
    """.

    Args:
        nnmodule: .
        optimizer_partial: .
        scheduler_partial: .
    """
    litmodule = BaseClasssificationLitModule(
        nnmodule,
        optimizer_partial,
        scheduler_partial,
        num_classes=3,
    )

    assert isinstance(litmodule.accuracy, MulticlassStatScores)
    assert litmodule.accuracy.num_classes == 3


@pytest.fixture()
def litmodule(
    nnmodule: nn.Module,
    optimizer_partial: partial[Optimizer],
    scheduler_partial: partial[LRScheduler],
) -> BaseClasssificationLitModule:
    """."""
    return BaseClasssificationLitModule(
        nnmodule,
        optimizer_partial,
        scheduler_partial,
        num_classes=3,
    )


def test_step(litmodule: BaseClasssificationLitModule) -> None:
    """.

    Args:
        litmodule: .
    """
    torch.manual_seed(0)

    (x, y) = (torch.randn(3, 5), torch.randint(high=3, size=(3,)))

    torch.manual_seed(0)

    loss = litmodule.step(
        batch=(x, y),
        stage="train",
    )

    torch.manual_seed(0)

    logits_test = litmodule.nnmodule(x)
    loss_test = torch.nn.functional.cross_entropy(logits_test, y)

    assert torch.isclose(loss, loss_test)
