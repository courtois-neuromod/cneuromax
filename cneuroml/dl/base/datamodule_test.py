"""Tests for Base DataModule.

Abbreviations:

PyTorch ``Dataset`` is short for ``torch.utils.data.Dataset``.

``Tensor`` is short for ``torch.Tensor``.

``Float`` is short for ``jaxtyping.Float``.
"""

import pytest
import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cneuroml.dl.base import BaseDataModule, BaseDataModuleConfig


def test_constructor_default_arguments() -> None:
    """Tests ``__init__`` method with mostly default config."""
    config = BaseDataModuleConfig(data_dir=".")
    datamodule = BaseDataModule(config)
    assert config == datamodule.config


def test_constructor_non_default_arguments() -> None:
    """Tests ``__init__`` method with non-default arguments."""
    config = BaseDataModuleConfig(
        data_dir="/",
        per_device_batch_size=2,
        per_device_num_workers=1,
        device_type="cpu",
    )
    datamodule = BaseDataModule(config)
    assert config == datamodule.config


@pytest.fixture()
def datamodule() -> BaseDataModule:
    """Creates and returns a generic ``BaseDataModule`` instance.

    Returns:
        A generic ``BaseDataModule`` instance.
    """
    config = BaseDataModuleConfig(data_dir=".")
    return BaseDataModule(config)


def test_load_state_dict(datamodule: BaseDataModule) -> None:
    """Tests ``load_state_dict`` method.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.
    """
    per_device_batch_size = 2

    datamodule.load_state_dict(
        {"per_device_batch_size": per_device_batch_size},
    )

    assert datamodule.config.per_device_batch_size == per_device_batch_size


@pytest.fixture()
def dataset() -> Dataset[Tensor]:
    """Creates and returns a generic PyTorch ``Dataset`` instance.

    Returns:
        A generic PyTorch ``Dataset`` instance.
    """

    class GenericDataset(Dataset[Tensor]):
        """A generic PyTorch dataset class.

        Attributes:
            data (``Tensor``): A PyTorch ``Tensor`` of shape ``(2, 1)``.
        """

        def __init__(self: "GenericDataset") -> None:
            """Initializes the dataset."""
            self.data = torch.zeros(2, 1)

        def __getitem__(
            self: "GenericDataset",
            index: int,
        ) -> Float[Tensor, "1"]:
            """Returns the data item at the given index.

            Args:
                index: The index of the item to return.

            Returns:
                The data item at the given index.
            """
            return self.data[index]

        def __len__(self: "GenericDataset") -> int:
            """Returns the length of the dataset.

            Returns:
                The length of the dataset.
            """
            return len(self.data)

    return GenericDataset()


@pytest.fixture()
def dataloader(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
) -> DataLoader[Tensor]:
    """Creates and returns a generic PyTorch ``DataLoader`` instance.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.
        dataset: A generic PyTorch ``Dataset`` instance.

    Returns:
        A generic PyTorch ``DataLoader`` instance.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=datamodule.config.per_device_batch_size,
        shuffle=True,
        num_workers=datamodule.config.per_device_num_workers,
        pin_memory=datamodule.config.pin_memory,
    )


def test_train_dataloader_with_data(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
    dataloader: DataLoader[Tensor],
) -> None:
    """Tests ``train_dataloader`` method behaviour with data.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.
        dataset: A generic PyTorch ``Dataset`` instance.
        dataloader: A generic PyTorch ``DataLoader`` instance.
    """
    datamodule.dataset["train"] = dataset
    new_dataloader = datamodule.train_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_train_dataloader_missing_data(datamodule: BaseDataModule) -> None:
    """Tests ``train_dataloader`` method behaviour missing data.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.

    Raises:
        AttributeError: If the instance's ``dataset["train"]`` attribute
            is not set.
    """
    with pytest.raises(AttributeError):
        datamodule.train_dataloader()


def test_val_dataloader_with_data(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
    dataloader: DataLoader[Tensor],
) -> None:
    """Tests ``val_dataloader`` method behaviour with data.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.
        dataset: A generic PyTorch ``Dataset`` instance.
        dataloader: A generic PyTorch ``DataLoader`` instance.
    """
    datamodule.dataset["val"] = dataset
    new_dataloader = datamodule.val_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_val_dataloader_missing_data(datamodule: BaseDataModule) -> None:
    """Tests ``val_dataloader`` method behaviour missing data.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.

    Raises:
        AttributeError: If ``dataset["val"]`` instance attribute is not
            set.
    """
    with pytest.raises(AttributeError):
        datamodule.val_dataloader()


def test_test_dataloader_with_data(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
    dataloader: DataLoader[Tensor],
) -> None:
    """Tests ``test_dataloader`` method behaviour with data.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.
        dataset: A generic PyTorch ``Dataset`` instance.
        dataloader: A generic PyTorch ``DataLoader`` instance.
    """
    datamodule.dataset["test"] = dataset
    new_dataloader = datamodule.test_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_test_dataloader_missing_data(datamodule: BaseDataModule) -> None:
    """Tests ``test_dataloader`` method behaviour missing data.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.

    Raises:
        AttributeError: If ``dataset["test"]`` instance attribute is not
            set.
    """
    with pytest.raises(AttributeError):
        datamodule.test_dataloader()


@pytest.fixture()
def dataloader_predict(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
) -> DataLoader[Tensor]:
    """Creates and returns a generic PyTorch ``DataLoader`` instance.

    This PyTorch ``DataLoader`` instance does not shuffle the dataset.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.
        dataset: A generic PyTorch ``Dataset`` instance.

    Returns:
        A generic PyTorch ``DataLoader`` instance.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=datamodule.config.per_device_batch_size,
        shuffle=False,
        num_workers=datamodule.config.per_device_num_workers,
        pin_memory=datamodule.config.pin_memory,
    )


def test_predict_dataloader_with_data(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
    dataloader_predict: DataLoader[Tensor],
) -> None:
    """Tests ``predict_dataloader`` method behaviour with data.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.
        dataset: A generic PyTorch ``Dataset`` instance.
        dataloader_predict: A generic PyTorch ``DataLoader`` instance.
    """
    datamodule.dataset["predict"] = dataset
    new_dataloader = datamodule.predict_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader_predict.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader_predict.sampler))
    assert new_dataloader.num_workers == dataloader_predict.num_workers
    assert new_dataloader.pin_memory == dataloader_predict.pin_memory


def test_predict_dataloader_missing_data(datamodule: BaseDataModule) -> None:
    """Tests ``predict_dataloader`` method behaviour missing data.

    Args:
        datamodule: A generic ``BaseDataModule`` instance.

    Raises:
        AttributeError: If ``dataset["predict"]`` instance attribute is
            not set.
    """
    with pytest.raises(AttributeError):
        datamodule.predict_dataloader()
