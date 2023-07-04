"""."""

from pathlib import Path

import pytest
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from cneuromax.deeplearning.common.datamodule.vision import (
    MNISTDataModule,
    MNISTDataModuleConfig,
)


@pytest.fixture()
def datamodule(tmp_path: Path) -> MNISTDataModule:
    """.

    Args:
        tmp_path: .

    Returns:
        A generic ``MNISTDataModule`` instance.
    """
    return MNISTDataModule(MNISTDataModuleConfig(data_dir=str(tmp_path) + "/"))


def test_setup_fit(datamodule: MNISTDataModule) -> None:
    """.

    Args:
        datamodule: .
    """
    datamodule.prepare_data()
    datamodule.setup("fit")

    assert isinstance(datamodule.dataset.train, Subset)
    assert isinstance(datamodule.dataset.val, Subset)

    assert len(datamodule.dataset.train.indices) == 54000
    assert len(datamodule.dataset.val.indices) == 6000


def test_setup_test(datamodule: MNISTDataModule) -> None:
    """.

    Args:
        datamodule: .
    """
    datamodule.prepare_data()
    datamodule.setup("test")

    assert isinstance(datamodule.dataset.test, MNIST)
    assert len(datamodule.dataset.test) == 10000
