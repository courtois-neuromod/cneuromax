"""Testing for base datamodule."""
from typing import Literal

import pytest
import torch
from jaxtyping import Float
from torch.utils.data import DataLoader, Dataset

from cneuroml.dl.base import BaseDataModule


def test_instantiate_generic_datamodule() -> None:
    """Testing instantiation with a generic datamodule."""
    datamodule = BaseDataModule(data_dir=".")

    assert datamodule.data_dir == "."
    assert datamodule.per_device_batch_size == 1
    assert datamodule.per_device_num_workers == 0
    assert datamodule.pin_memory


def test_instantiate_datamodule_non_default_arguments() -> None:
    """Testing instantiation with non-default arguments."""
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
    """Instantiates and returns a generic BaseDataModule.

    Returns:
        A generic BaseDataModule.
    """
    return BaseDataModule(data_dir=".")


def test_load_state_dict(datamodule: BaseDataModule) -> None:
    """Test load_state_dict.

    Args:
        datamodule: A generic datamodule.
    """
    per_device_batch_size = 2

    datamodule.load_state_dict(
        {"per_device_batch_size": per_device_batch_size},
    )

    assert datamodule.per_device_batch_size == per_device_batch_size


def test_state_dict(datamodule: BaseDataModule) -> None:
    """Test state_dict.

    Args:
        datamodule: A generic datamodule.
    """
    assert (
        datamodule.state_dict()["per_device_batch_size"]
        == datamodule.per_device_batch_size
    )


@pytest.fixture()
def dataset() -> Dataset[torch.Tensor]:
    """Instantiates and returns a generic PyTorch Dataset.

    Returns:
        A generic PyTorch dataset.
    """

    class GenericDataset(Dataset[torch.Tensor]):
        """A generic PyTorch dataset.

        Attributes:
            data: The data.
        """

        def __init__(self: "GenericDataset") -> None:
            """Initializes the dataset."""
            self.data = torch.zeros(1, 1)

        def __getitem__(
            self: "GenericDataset",
            index: int,
        ) -> Float[torch.Tensor, "1"]:
            """Returns the item at the given index.

            Args:
                index: The index of the item to return.

            Returns:
                The item at the given index.
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
    dataset: Dataset[torch.Tensor],
) -> DataLoader[torch.Tensor]:
    """Instantiates and returns a generic PyTorch DataLoader.

    Args:
        datamodule: A generic datamodule.
        dataset: A generic dataset.

    Returns:
        A generic PyTorch DataLoader.
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
    """Test 'train_dataloader' behaviour.

    Args:
        datamodule: A generic datamodule.
        dataset: A generic dataset with random values.
        dataloader: A generic dataloader.
    """
    datamodule.train_data = dataset
    new_dataloader = datamodule.train_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_val_dataloader_correct_values(
    datamodule: BaseDataModule,
    dataset: Dataset[torch.Tensor],
    dataloader: DataLoader[torch.Tensor],
) -> None:
    """Test 'val_dataloader' behaviour.

    Args:
        datamodule: A generic datamodule.
        dataset: A generic dataset with random values.
        dataloader: A generic dataloader.
    """
    datamodule.val_data = dataset
    new_dataloader = datamodule.val_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_test_dataloader_correct_values(
    datamodule: BaseDataModule,
    dataset: Dataset[torch.Tensor],
    dataloader: DataLoader[torch.Tensor],
) -> None:
    """Test 'test_dataloader' behaviour.

    Args:
        datamodule: A generic datamodule.
        dataset: A generic dataset with random values.
        dataloader: A generic dataloader.
    """
    datamodule.test_data = dataset
    new_dataloader = datamodule.test_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


@pytest.fixture()
def dataloader2(
    datamodule: BaseDataModule,
    dataset: Dataset[torch.Tensor],
) -> DataLoader[torch.Tensor]:
    """Instantiates and returns a generic PyTorch DataLoader.

    Args:
        datamodule: A generic datamodule.
        dataset: A generic dataset.

    Returns:
        A generic PyTorch DataLoader.
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
    dataloader2: DataLoader[torch.Tensor],
) -> None:
    """Test 'predict_dataloader' behaviour.

    Args:
        datamodule: A generic datamodule.
        dataset: A generic dataset with random values.
        dataloader2: A generic dataloader.
    """
    datamodule.predict_data = dataset
    new_dataloader = datamodule.predict_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader2.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader2.sampler))
    assert new_dataloader.num_workers == dataloader2.num_workers
    assert new_dataloader.pin_memory == dataloader2.pin_memory
