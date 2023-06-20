"""Test for base datamodule."""
from typing import Literal

import pytest
import torch
from jaxtyping import Float
from torch.utils.data import DataLoader, Dataset

from cneuroml.dl.base import BaseDataModule


def test_instantiate_datamodule_default_arguments() -> None:
    """Test instantiation with default arguments."""
    datamodule = BaseDataModule(data_dir=".")

    assert datamodule.data_dir == "."
    assert datamodule.per_device_batch_size == 1
    assert datamodule.per_device_num_workers == 0
    assert datamodule.pin_memory


def test_instantiate_datamodule_non_default_arguments() -> None:
    """Test instantiation with non-default arguments."""
    data_dir = "/"
    per_device_batch_size = 2
    per_device_num_workers = 1
    device_type: Literal["cpu", "gpu"] = "cpu"

    datamodule = BaseDataModule(
        data_dir=data_dir,
        per_device_batch_size=per_device_batch_size,
        per_device_num_workers=per_device_num_workers,
        device_type=device_type,
    )

    assert datamodule.data_dir == data_dir
    assert datamodule.per_device_batch_size == per_device_batch_size
    assert datamodule.per_device_num_workers == per_device_num_workers
    assert not datamodule.pin_memory


@pytest.fixture()
def datamodule() -> BaseDataModule:
    """Instantiates and returns a generic ``BaseDataModule`` object.

    Returns:
        A generic ``BaseDataModule`` object.
    """
    return BaseDataModule(data_dir=".")


def test_load_state_dict(datamodule: BaseDataModule) -> None:
    """Test ``load_state_dict`` method.

    Args:
        datamodule: A generic ``BaseDataModule`` object.
    """
    per_device_batch_size = 2

    datamodule.load_state_dict(
        {"per_device_batch_size": per_device_batch_size},
    )

    assert datamodule.per_device_batch_size == per_device_batch_size


def test_state_dict(datamodule: BaseDataModule) -> None:
    """Test ``state_dict`` method.

    Args:
        datamodule: A generic ``BaseDataModule`` object.
    """
    assert (
        datamodule.state_dict()["per_device_batch_size"]
        == datamodule.per_device_batch_size
    )


@pytest.fixture()
def dataset() -> Dataset[torch.Tensor]:
    """Instantiates and returns a generic PyTorch ``Dataset`` object.

    Returns:
        A generic ``torch.utils.data.Dataset`` object.
    """

    class GenericDataset(Dataset[torch.Tensor]):
        """A generic PyTorch dataset class.

        Attributes:
            data (``torch.Tensor``): A tensor of shape ``(1, 1)``.
        """

        def __init__(self: "GenericDataset") -> None:
            """Initializes the dataset."""
            self.data = torch.zeros(1, 1)

        def __getitem__(
            self: "GenericDataset",
            index: int,
        ) -> Float[torch.Tensor, "1"]:
            """Returns the data item at the given index.

            Args:
                index: The index of the item to return.

            Returns:
                The data item at the given index.
            """
            return self.data[index]

        def __len__(self: "GenericDataset") -> int:
            """Returns the length of this dataset.

            Returns:
                The length of this dataset.
            """
            return len(self.data)

    return GenericDataset()


@pytest.fixture()
def dataloader(
    datamodule: BaseDataModule,
    dataset: Dataset[torch.Tensor],
) -> DataLoader[torch.Tensor]:
    """Instantiates and returns a generic PyTorch ``DataLoader`` object.

    Args:
        datamodule: A generic ``BaseDataModule`` object.
        dataset: A generic ``torch.utils.data.Dataset`` object.

    Returns:
        A generic ``torch.utils.data.DataLoader`` object.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=datamodule.per_device_batch_size,
        shuffle=True,
        num_workers=datamodule.per_device_num_workers,
        pin_memory=datamodule.pin_memory,
    )


def test_train_dataloader_correct_values(
    datamodule: BaseDataModule,
    dataset: Dataset[torch.Tensor],
    dataloader: DataLoader[torch.Tensor],
) -> None:
    """Test ``train_dataloader`` method behaviour.

    Args:
        datamodule: A generic ``BaseDataModule`` object.
        dataset: A generic ``torch.utils.data.Dataset`` object.
        dataloader: A generic ``torch.utils.data.DataLoader`` object.
    """
    datamodule.train_data = dataset
    new_dataloader = datamodule.train_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_train_dataloader_missing_data(datamodule: BaseDataModule) -> None:
    """Test ``train_dataloader`` method behaviour with missing data.

    Missing ``train_data`` attribute should raise an ``AttributeError``
    exception.

    Args:
        datamodule: A generic ``BaseDataModule`` object.

    Raises:
        AttributeError: If ``train_data`` is not set.
    """
    with pytest.raises(AttributeError):
        datamodule.train_dataloader()


def test_val_dataloader_correct_values(
    datamodule: BaseDataModule,
    dataset: Dataset[torch.Tensor],
    dataloader: DataLoader[torch.Tensor],
) -> None:
    """Test ``val_dataloader`` method behaviour.

    Args:
        datamodule: A generic ``BaseDataModule`` object.
        dataset: A generic ``torch.utils.data.Dataset`` object.
        dataloader: A generic ``torch.utils.data.DataLoader`` object.
    """
    datamodule.val_data = dataset
    new_dataloader = datamodule.val_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_val_dataloader_missing_data(datamodule: BaseDataModule) -> None:
    """Test ``val_dataloader`` behaviour with missing attribute.

    Missing ``val_data`` attribute should raise an ``AttributeError``
    exception.

    Args:
        datamodule: A generic datamodule.

    Raises:
        AttributeError: If ``val_data`` is not set.
    """
    with pytest.raises(AttributeError):
        datamodule.val_dataloader()


def test_test_dataloader_correct_values(
    datamodule: BaseDataModule,
    dataset: Dataset[torch.Tensor],
    dataloader: DataLoader[torch.Tensor],
) -> None:
    """Test ``test_dataloader`` behaviour.

    Args:
        datamodule: A generic ``BaseDataModule`` object.
        dataset: A generic ``torch.utils.data.Dataset`` object.
        dataloader: A generic ``torch.utils.data.DataLoader`` object.
    """
    datamodule.test_data = dataset
    new_dataloader = datamodule.test_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_test_dataloader_missing_data(datamodule: BaseDataModule) -> None:
    """Test ``test_dataloader`` behaviour with missing attribute.

    Missing ``test_data`` attribute should raise an ``AttributeError``
    exception.

    Args:
        datamodule: A generic datamodule.

    Raises:
        AttributeError: If ``test_data`` is not set.
    """
    with pytest.raises(AttributeError):
        datamodule.test_dataloader()


@pytest.fixture()
def dataloader_predict(
    datamodule: BaseDataModule,
    dataset: Dataset[torch.Tensor],
) -> DataLoader[torch.Tensor]:
    """Instantiates and returns a generic PyTorch ``DataLoader`` object.

    Args:
        datamodule: A generic ``BaseDataModule`` object.
        dataset: A generic ``torch.utils.data.Dataset`` object.

    Returns:
        A generic ``torch.utils.data.DataLoader`` object.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=datamodule.per_device_batch_size,
        shuffle=False,
        num_workers=datamodule.per_device_num_workers,
        pin_memory=datamodule.pin_memory,
    )


def test_predict_dataloader_correct_values(
    datamodule: BaseDataModule,
    dataset: Dataset[torch.Tensor],
    dataloader_predict: DataLoader[torch.Tensor],
) -> None:
    """Test ``predict_dataloader`` behaviour.

    Args:
        datamodule: A generic ``BaseDataModule`` object.
        dataset: A generic ``torch.utils.data.Dataset`` object.
        dataloader_predict: A generic ``torch.utils.data.DataLoader``
            object.
    """
    datamodule.predict_data = dataset
    new_dataloader = datamodule.predict_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader_predict.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader_predict.sampler))
    assert new_dataloader.num_workers == dataloader_predict.num_workers
    assert new_dataloader.pin_memory == dataloader_predict.pin_memory


def test_predict_dataloader_missing_data(datamodule: BaseDataModule) -> None:
    """Test ``predict_dataloader`` method behaviour with missing data.

    Missing ``predict_data`` attribute should raise an
    ``AttributeError`` exception.

    Args:
        datamodule: A generic ``BaseDataModule`` object.

    Raises:
        AttributeError: If ``predict_data`` is not set.
    """
    with pytest.raises(AttributeError):
        datamodule.predict_dataloader()
