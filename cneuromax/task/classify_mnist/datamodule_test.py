"""Tests for :mod:`~cneuromax.task.classify_mnist.datamodule`."""

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
    """:class:`~.MNISTClassificationDataModule` fixture.

    Args:
        tmp_path: The temporary path for the\
            :class:`~.MNISTClassificationDataModule`.

    Returns:
        A generic :class:`~.MNISTClassificationDataModule` instance.
    """
    return MNISTClassificationDataModule(
        MNISTClassificationDataModuleConfig(
            data_dir=str(tmp_path) + "/",
            device="cpu",
            val_percentage=0.1,
        ),
    )


def test_setup_fit(datamodule: MNISTClassificationDataModule) -> None:
    """Tests :meth:`~.MNISTClassificationDataModule.setup` #1.

    Args:
        datamodule: A generic :class:`~.MNISTClassificationDataModule`\
            instance, see :func:`datamodule`.
    """
    datamodule.prepare_data()
    datamodule.setup("fit")

    assert isinstance(datamodule.dataset.train, Subset)
    assert isinstance(datamodule.dataset.val, Subset)

    assert len(datamodule.dataset.train) == 54000
    assert len(datamodule.dataset.val) == 6000


def test_setup_test(datamodule: MNISTClassificationDataModule) -> None:
    """Tests :meth:`~.MNISTClassificationDataModule.setup` #2.

    Args:
        datamodule: A generic :class:`~.MNISTClassificationDataModule`\
            instance, see :func:`datamodule`.
    """
    datamodule.prepare_data()
    datamodule.setup("test")

    assert isinstance(datamodule.dataset.test, MNIST)
    assert len(datamodule.dataset.test) == 10000
