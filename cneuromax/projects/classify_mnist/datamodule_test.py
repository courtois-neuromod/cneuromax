""":mod:`~.classify_mnist.datamodule` tests."""

from pathlib import Path

import pytest
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from . import (
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

    Verifies that :func:`~.MNISTClassificationDataModule.setup` behaves
    correctly when
    :paramref:`~.MNISTClassificationDataModule.setup.stage` is
    ``"fit"``.

    Args:
        datamodule: A generic :class:`~.MNISTClassificationDataModule`\
            instance, see :func:`datamodule`.
    """
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    assert isinstance(datamodule.datasets.train, Subset)
    assert isinstance(datamodule.datasets.val, Subset)

    assert len(datamodule.datasets.train) == 54000
    assert len(datamodule.datasets.val) == 6000


def test_setup_test(datamodule: MNISTClassificationDataModule) -> None:
    """Tests :meth:`~.MNISTClassificationDataModule.setup` #2.

    Verifies that :func:`~.MNISTClassificationDataModule.setup` behaves
    correctly when
    :paramref:`~.MNISTClassificationDataModule.setup.stage` is
    ``"test"``.

    Args:
        datamodule: A generic :class:`~.MNISTClassificationDataModule`\
            instance, see :func:`datamodule`.
    """
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    assert isinstance(datamodule.datasets.test, MNIST)
    assert len(datamodule.datasets.test) == 10000
