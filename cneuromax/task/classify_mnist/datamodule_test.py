"""."""

from pathlib import Path

import pytest
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from cneuromax.task.classify_mnist import (
    MNISTClassificationDataModule,
    MNISTClassificationDataModuleConfig,
)


@pytest.fixture()
def datamodule(tmp_path: Path) -> MNISTClassificationDataModule:
    """.

    Args:
        tmp_path: .

    Returns:
        A generic ``MNISTDataModule`` instance.
    """
    return MNISTClassificationDataModule(
        MNISTClassificationDataModuleConfig(
            data_dir=str(tmp_path) + "/",
            device="cpu",
            val_percentage=0.1,
        ),
    )


def test_setup_fit(datamodule: MNISTClassificationDataModule) -> None:
    """.

    Args:
        datamodule: .
    """
    datamodule.prepare_data()
    datamodule.setup("fit")

    assert isinstance(datamodule.dataset.train, Subset)
    assert isinstance(datamodule.dataset.val, Subset)

    assert len(datamodule.dataset.train) == 54000
    assert len(datamodule.dataset.val) == 6000


def test_setup_test(datamodule: MNISTClassificationDataModule) -> None:
    """.

    Args:
        datamodule: .
    """
    datamodule.prepare_data()
    datamodule.setup("test")

    assert isinstance(datamodule.dataset.test, MNIST)
    assert len(datamodule.dataset.test) == 10000
