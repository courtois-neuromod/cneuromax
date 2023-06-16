"""Base DataModule.

Abbreviations:

Lightning's ``LightningDataModule`` is short for
``lightning.pytorch.LightningDataModule``.

PyTorch ``Dataset`` is short for ``torch.utils.data.Dataset``.

PyTorch ``DataLoader`` is short for ``torch.utils.data.DataLoader``.

``Tensor`` is short for ``torch.Tensor``.
"""

from abc import ABCMeta
from dataclasses import dataclass
from typing import Literal, final

from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass
class BaseDataModuleConfig:
    """Base DataModule configuration dataclass.

    Stores the configuration for a ``BaseDataModule`` instance.

    Attributes:
        data_dir: Path to the data directory.
        per_device_batch_size: Per-device number of samples to load
            per iteration.
        per_device_num_workers: Per-device number of CPU processes to
            use for data loading (``0`` means that the data will be
            loaded by each device's assigned CPU process)
        device_type: The compute device type to use (``cpu`` or
            ``gpu``).
        pin_memory: Whether to copy tensors into device pinned
            memory before returning them (is set to ``True`` by default
            if using GPUs).
    """

    data_dir: str
    per_device_batch_size: int = 1
    per_device_num_workers: int = 0
    device_type: Literal["cpu", "gpu"] = "gpu"
    pin_memory: bool = device_type == "gpu"


class BaseDataModule(LightningDataModule, metaclass=ABCMeta):
    """The Base DataModule class.

    This class inherits from Lightning's ``LightningDataModule`` class.
    With ``stage`` being any of ``"train"``, ``"val"``, ``"test"`` or
    ``"predict"``, children of this class need to properly define
    ``dataset[stage]`` instance attribute(s) for each desired ``stage``.

    Attributes:
        config (``BaseDataModuleConfig``): The base dataclass
            configuration instance.
        dataset (``dict[Literal["train", "val", "test", "predict"],
            Dataset]``): The dataset dictionary containing the PyTorch
            ``Dataset`` instance(s) for each desired stage.
    """

    def __init__(
        self: "BaseDataModule",
        config: BaseDataModuleConfig,
    ) -> None:
        """Constructor.

        Calls parent constructor and stores the ``config`` and
        type-hints ``dataset`` instance attributes.

        Args:
            config: A ``BaseDataModuleConfig`` instance.
        """
        super().__init__()
        self.config = config
        self.dataset: dict[
            Literal["train", "val", "test", "predict"],
            Dataset[Tensor],
        ] = {}

    @final
    def load_state_dict(
        self: "BaseDataModule",
        state_dict: dict[str, int],
    ) -> None:
        """Self-assigns an existing ``state_dict``.

        Currently this is only used to set the instance's
        ``config.per_device_batch_size``.

        Args:
            state_dict: A state dictionary to self-assign.
        """
        self.config.per_device_batch_size = state_dict["per_device_batch_size"]

    @final
    def state_dict(self: "BaseDataModule") -> dict[str, int]:
        """Returns the instance's ``config.state_dict``.

        Currently this is only used to return the instance's
        ``config.per_device_batch_size``.

        Returns:
            A copy of the instance's state dictionary.
        """
        return {"per_device_batch_size": self.config.per_device_batch_size}

    @final
    def train_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Returns a new training PyTorch ``DataLoader`` instance.

        Creates and returns a new PyTorch ``DataLoader`` instance built
        using the instance's ``dataset["train"]`` and ``config``
        attributes.

        Returns:
            A new training PyTorch ``DataLoader`` instance.

        Raises:
            AttributeError: If the instance's ``dataset["train"]``
            attribute is not defined.
        """
        if "train" not in self.dataset:
            raise AttributeError

        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.config.per_device_batch_size,
            shuffle=True,
            num_workers=self.config.per_device_num_workers,
            pin_memory=self.config.pin_memory,
        )

    @final
    def val_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Returns a new validation PyTorch ``DataLoader`` instance.

        Creates and returns a new PyTorch ``DataLoader`` instance built
        using the instance's ``dataset["val"]`` and ``config``
        attributes.

        Returns:
            A new validation PyTorch ``DataLoader`` instance.

        Raises:
            AttributeError: If the instance's ``dataset["val"]``
            attribute is not defined.
        """
        if "val" not in self.dataset:
            raise AttributeError

        return DataLoader(
            dataset=self.dataset["val"],
            batch_size=self.config.per_device_batch_size,
            shuffle=True,
            num_workers=self.config.per_device_num_workers,
            pin_memory=self.config.pin_memory,
        )

    @final
    def test_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Returns a new testing PyTorch ``DataLoader`` instance.

        Creates and returns a new PyTorch ``DataLoader`` instance built
        using the instance's ``dataset["test"]`` and ``config``
        attributes.

        Returns:
            A new testing PyTorch ``DataLoader`` instance.

        Raises:
            AttributeError: If the instance's ``dataset["test"]``
            attribute is not defined.
        """
        if "test" not in self.dataset:
            raise AttributeError

        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.config.per_device_batch_size,
            shuffle=True,
            num_workers=self.config.per_device_num_workers,
            pin_memory=self.config.pin_memory,
        )

    @final
    def predict_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Returns a new prediction PyTorch ``DataLoader`` instance.

        Creates and returns a new PyTorch ``DataLoader`` instance built
        using the instance's ``dataset["predict"]`` and ``config``
        attributes. The new PyTorch ``DataLoader`` instance does not
        shuffle the dataset.

        Returns:
            A new prediction PyTorch ``DataLoader`` instance.

        Raises:
            AttributeError: If the instance's ``dataset["predict"]``
            attribute is not defined.
        """
        if "predict" not in self.dataset:
            raise AttributeError

        return DataLoader(
            dataset=self.dataset["predict"],
            batch_size=self.config.per_device_batch_size,
            shuffle=False,
            num_workers=self.config.per_device_num_workers,
            pin_memory=self.config.pin_memory,
        )
