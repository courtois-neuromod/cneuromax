"""."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from cneuroml.deeplearning.common.datamodule import BaseDataModuleConfig
from cneuroml.deeplearning.common.datamodule.vision import (
    MNISTDataModule,
)


@pytest.fixture()
def config(tmp_path: Path) -> BaseDataModuleConfig:
    """.

    Returns:
        A generic ``BaseDataModuleConfig`` instance.
    """
    return BaseDataModuleConfig(data_dir=str(tmp_path) + "/")


def test_constructor(config: BaseDataModuleConfig) -> None:
    """.

    Args:
        config: .
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
        ],
    )

    mnist_data_module = MNISTDataModule(
        config,
        val_percentage=0.24,
        transform=transform,
    )
    assert mnist_data_module.train_val_split == [0.76, 0.24]
    assert mnist_data_module.transform == transform


@pytest.fixture()
def datamodule(config: BaseDataModuleConfig) -> MNISTDataModule:
    """.

    Args:
        config: .

    Returns:
        A generic ``MNISTDataModule`` instance.
    """
    return MNISTDataModule(config)


def test_prepare_data(datamodule: MNISTDataModule) -> None:
    """.

    Args:
        datamodule: .
    """
    datamodule.prepare_data()

    assert Path(
        datamodule.config.data_dir + "/MNIST/raw/t10k-images-idx3-ubyte",
    ).exists()

    assert Path(
        datamodule.config.data_dir + "/MNIST/raw/t10k-images-idx3-ubyte.gz",
    ).exists()

    assert Path(
        datamodule.config.data_dir + "/MNIST/raw/t10k-labels-idx1-ubyte",
    ).exists()

    assert Path(
        datamodule.config.data_dir + "/MNIST/raw/t10k-labels-idx1-ubyte.gz",
    ).exists()

    assert Path(
        datamodule.config.data_dir + "/MNIST/raw/train-images-idx3-ubyte",
    ).exists()

    assert Path(
        datamodule.config.data_dir + "/MNIST/raw/train-images-idx3-ubyte.gz",
    ).exists()

    assert Path(
        datamodule.config.data_dir + "/MNIST/raw/train-labels-idx1-ubyte",
    ).exists()

    assert Path(
        datamodule.config.data_dir + "/MNIST/raw/train-labels-idx1-ubyte.gz",
    ).exists()


def test_setup_fit(datamodule: MNISTDataModule) -> None:
    """.

    Args:
        datamodule: .
    """
    datamodule.prepare_data()
    datamodule.setup("fit")

    assert isinstance(datamodule.dataset["train"], Subset)
    assert isinstance(datamodule.dataset["val"], Subset)

    assert (
        datamodule.dataset["train"].dataset
        == datamodule.dataset["val"].dataset
    )
    assert isinstance(datamodule.dataset["train"].dataset, MNIST)
    assert torch.isclose(
        datamodule.dataset["train"].dataset.data.float().mean(),
        torch.tensor(33.3184),
    )
    assert torch.isclose(
        datamodule.dataset["train"].dataset.data.float().std(),
        torch.tensor(78.5675),
    )
    assert len(datamodule.dataset["train"].indices) == 54000
    assert len(datamodule.dataset["val"].indices) == 6000
    assert datamodule.dataset["train"].dataset.train is True
    assert (
        datamodule.dataset["train"].dataset.transform == datamodule.transform
    )


def test_setup_test(datamodule: MNISTDataModule) -> None:
    """.

    Args:
        datamodule: .
    """
    datamodule.prepare_data()
    datamodule.setup("test")

    assert isinstance(datamodule.dataset["test"], MNIST)
    assert torch.isclose(
        datamodule.dataset["test"].data.float().mean(),
        torch.tensor(33.7912),
    )
    assert torch.isclose(
        datamodule.dataset["test"].data.float().std(),
        torch.tensor(79.1725),
    )
    assert len(datamodule.dataset["test"]) == 10000
    assert datamodule.dataset["test"].train is False
    assert datamodule.dataset["test"].transform == datamodule.transform
