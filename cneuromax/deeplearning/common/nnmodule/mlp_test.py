"""."""

import pytest
import torch
from einops import rearrange
from torch import nn

from cneuromax.deeplearning.common.nnmodule import MLP, MLPConfig


@pytest.fixture()
def mlp_1() -> MLP:
    """."""
    config = MLPConfig(dims=[1, 2, 3])
    return MLP(config)


@pytest.fixture()
def mlp_2() -> MLP:
    """."""
    config = MLPConfig(
        dims=[2, 3, 4, 5],
        activation_fn=nn.Sigmoid,
        p_dropout=0.5,
    )
    return MLP(config)


def test_constructor(mlp_1: MLP, mlp_2: MLP) -> None:
    """Tests two different MLPs.

    Args:
        mlp_1: .
        mlp_2: .
    """
    assert isinstance(mlp_1.model, nn.Sequential)

    assert len(mlp_1.model) == 3
    assert isinstance(mlp_1.model[0], nn.Linear)
    assert mlp_1.model[0].in_features == 1
    assert mlp_1.model[0].out_features == 2
    assert isinstance(mlp_1.model[1], nn.ReLU)
    assert isinstance(mlp_1.model[2], nn.Linear)
    assert mlp_1.model[2].in_features == 2
    assert mlp_1.model[2].out_features == 3

    assert len(mlp_2.model) == 7
    assert isinstance(mlp_2.model[0], nn.Linear)
    assert mlp_2.model[0].in_features == 2
    assert mlp_2.model[0].out_features == 3
    assert isinstance(mlp_2.model[1], nn.Sigmoid)
    assert isinstance(mlp_2.model[2], nn.Dropout)
    assert mlp_2.model[2].p == 0.5
    assert isinstance(mlp_2.model[3], nn.Linear)
    assert mlp_2.model[3].in_features == 3
    assert mlp_2.model[3].out_features == 4
    assert isinstance(mlp_2.model[4], nn.Sigmoid)
    assert isinstance(mlp_2.model[5], nn.Dropout)
    assert mlp_2.model[5].p == 0.5
    assert isinstance(mlp_2.model[6], nn.Linear)
    assert mlp_2.model[6].in_features == 4
    assert mlp_2.model[6].out_features == 5


def test_forward(mlp_1: MLP, mlp_2: MLP) -> None:
    """.

    Args:
        mlp_1: .
        mlp_2: .
    """
    x = torch.randn(10, 1)
    torch.manual_seed(0)
    out = mlp_1(x)
    torch.manual_seed(0)
    x = rearrange(x, "batch_size ... -> batch_size (...)")
    out_expected = mlp_1.model(x)
    assert torch.allclose(out, out_expected)

    x = torch.randn(10, 2)
    torch.manual_seed(0)
    out = mlp_2(x)
    torch.manual_seed(0)
    x = rearrange(x, "batch_size ... -> batch_size (...)")
    out_expected = mlp_2.model(x)
    assert torch.allclose(out, out_expected)
