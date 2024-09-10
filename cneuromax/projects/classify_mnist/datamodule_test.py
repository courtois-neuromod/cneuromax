""":mod:`~.classify_mnist.datamodule` tests."""

from pathlib import Path

import pytest
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from . import (
    MNISTDataModule,
    MNISTDataModuleConfig,
)


@pytest.fixture
def datamodule(tmp_path: Path) -> MNISTDataModule:
    """:class:`~.MNISTDataModule` fixture.

    Args:
        tmp_path: The temporary path for the :class:`~.MNISTDataModule`.

    Returns:
        A generic :class:`~.MNISTDataModule` instance.
    """
    return MNISTDataModule(
        MNISTDataModuleConfig(
            data_dir=str(tmp_path) + "/",
            device="cpu",
            val_percentage=0.1,
        ),
    )


def test_setup_fit(datamodule: MNISTDataModule) -> None:
    """Tests :meth:`~.MNISTDataModule.setup` #1.

    Verifies that :func:`~.MNISTDataModule.setup` behaves
    correctly when :paramref:`~.MNISTDataModule.setup.stage` is
    ``"fit"``.

    Args:
        datamodule: A generic :class:`~.MNISTDataModule`
            instance, see :func:`datamodule`.
    """
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    assert isinstance(datamodule.datasets.train, Subset)
    assert isinstance(datamodule.datasets.val, Subset)

    assert len(datamodule.datasets.train) == 54000
    assert len(datamodule.datasets.val) == 6000


def test_setup_test(datamodule: MNISTDataModule) -> None:
    """Tests :meth:`~.MNISTDataModule.setup` #2.

    Verifies that :func:`~.MNISTDataModule.setup` behaves
    correctly when :paramref:`~.MNISTDataModule.setup.stage` is
    ``"test"``.

    Args:
        datamodule: A generic :class:`~.MNISTDataModule`
            instance, see :func:`datamodule`.
    """
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    assert isinstance(datamodule.datasets.test, MNIST)
    assert len(datamodule.datasets.test) == 10000
