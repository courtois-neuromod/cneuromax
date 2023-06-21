"""."""

import pytest
import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cneuroml.deeplearning.common.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)


def test_constructor_default_arguments() -> None:
    """."""
    config = BaseDataModuleConfig(data_dir=".")
    datamodule = BaseDataModule(config)
    assert config == datamodule.config


def test_constructor_non_default_arguments() -> None:
    """."""
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
    """.

    Returns:
        A generic ``BaseDataModule`` instance.
    """
    config = BaseDataModuleConfig(data_dir=".")
    return BaseDataModule(config)


def test_load_state_dict(datamodule: BaseDataModule) -> None:
    """.

    Args:
        datamodule: .
    """
    per_device_batch_size = 2

    datamodule.load_state_dict(
        {"per_device_batch_size": per_device_batch_size},
    )

    assert datamodule.config.per_device_batch_size == per_device_batch_size


@pytest.fixture()
def dataset() -> Dataset[Tensor]:
    """.

    Returns:
        A generic PyTorch ``Dataset`` instance.
    """

    class GenericDataset(Dataset[Tensor]):
        """.

        Attributes:
            data (``Tensor``): .
        """

        def __init__(self: "GenericDataset") -> None:
            """."""
            self.data = torch.zeros(2, 1)

        def __getitem__(
            self: "GenericDataset",
            index: int,
        ) -> Float[Tensor, "1"]:
            """.

            Returns:
                .
            """
            return self.data[index]

        def __len__(self: "GenericDataset") -> int:
            """.

            Returns:
                .
            """
            return len(self.data)

    return GenericDataset()


@pytest.fixture()
def dataloader(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
) -> DataLoader[Tensor]:
    """.

    Args:
        datamodule: .
        dataset: .

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


def test_train_dataloader_with_required_dataset_attribute(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
    dataloader: DataLoader[Tensor],
) -> None:
    """.

    Args:
        datamodule: .
        dataset: .
        dataloader: .
    """
    datamodule.dataset["train"] = dataset
    new_dataloader = datamodule.train_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_train_dataloader_with_missing_dataset_attribute(
    datamodule: BaseDataModule,
) -> None:
    """.

    Args:
        datamodule: .
    """
    with pytest.raises(AttributeError):
        datamodule.train_dataloader()


def test_val_dataloader_with_required_dataset_attribute(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
    dataloader: DataLoader[Tensor],
) -> None:
    """.

    Args:
        datamodule: .
        dataset: .
        dataloader: .
    """
    datamodule.dataset["val"] = dataset
    new_dataloader = datamodule.val_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_val_dataloader_with_missing_dataset_attribute(
    datamodule: BaseDataModule,
) -> None:
    """.

    Args:
        datamodule: .
    """
    with pytest.raises(AttributeError):
        datamodule.val_dataloader()


def test_test_dataloader_with_required_dataset_attribute(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
    dataloader: DataLoader[Tensor],
) -> None:
    """.

    Args:
        datamodule: .
        dataset: .
        dataloader: .
    """
    datamodule.dataset["test"] = dataset
    new_dataloader = datamodule.test_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(dataloader.sampler))
    assert new_dataloader.num_workers == dataloader.num_workers
    assert new_dataloader.pin_memory == dataloader.pin_memory


def test_test_dataloader_with_missing_dataset_attribute(
    datamodule: BaseDataModule,
) -> None:
    """.

    Args:
        datamodule: .
    """
    with pytest.raises(AttributeError):
        datamodule.test_dataloader()


@pytest.fixture()
def predict_dataloader(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
) -> DataLoader[Tensor]:
    """Returns a ``DataLoader``that does not shuffle the dataset.

    Args:
        datamodule: .
        dataset: .

    Returns:
        .
    """
    return DataLoader(
        dataset=dataset,
        batch_size=datamodule.config.per_device_batch_size,
        shuffle=False,
        num_workers=datamodule.config.per_device_num_workers,
        pin_memory=datamodule.config.pin_memory,
    )


def test_predict_dataloader_with_required_dataset_attribute(
    datamodule: BaseDataModule,
    dataset: Dataset[Tensor],
    predict_dataloader: DataLoader[Tensor],
) -> None:
    """.

    Args:
        datamodule: .
        dataset: .
        predict_dataloader: .
    """
    datamodule.dataset["predict"] = dataset
    new_dataloader = datamodule.predict_dataloader()
    assert isinstance(new_dataloader, DataLoader)
    assert new_dataloader.batch_size == predict_dataloader.batch_size
    assert isinstance(new_dataloader.sampler, type(predict_dataloader.sampler))
    assert new_dataloader.num_workers == predict_dataloader.num_workers
    assert new_dataloader.pin_memory == predict_dataloader.pin_memory


def test_predict_dataloader_with_missing_dataset_attribute(
    datamodule: BaseDataModule,
) -> None:
    """.

    Args:
        datamodule: .
    """
    with pytest.raises(AttributeError):
        datamodule.predict_dataloader()
