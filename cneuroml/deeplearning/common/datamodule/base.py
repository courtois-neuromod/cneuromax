"""."""

from abc import ABCMeta
from dataclasses import dataclass
from typing import Literal, final

from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass
class BaseDataModuleConfig:
    """.

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
    """.

    With ``stage`` being any of ``"train"``, ``"val"``, ``"test"`` or
    ``"predict"``, children of this class need to properly define
    ``dataset[stage]`` instance attribute(s) for each desired ``stage``.

    Attributes:
        config (``BaseDataModuleConfig``): .
        dataset (``dict[Literal["train", "val", "test", "predict"],
            Dataset]``): .
    """

    def __init__(self: "BaseDataModule", config: BaseDataModuleConfig) -> None:
        """Calls parent constructor.

        Args:
            config: .
        """
        super().__init__()
        self.config: BaseDataModuleConfig = config
        self.dataset: dict[  # For static type checking
            Literal["train", "val", "test", "predict"],
            Dataset[Tensor],
        ] = {}

    @final
    def load_state_dict(
        self: "BaseDataModule",
        state_dict: dict[str, int],
    ) -> None:
        """Sets the instance's batch size from a dictionary value.

        Args:
            state_dict: .
        """
        self.config.per_device_batch_size = state_dict["per_device_batch_size"]

    @final
    def state_dict(self: "BaseDataModule") -> dict[str, int]:
        """.

        Returns:
            The instance's batch size inside a dictionary.
        """
        return {"per_device_batch_size": self.config.per_device_batch_size}

    @final
    def train_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """.

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
        """.

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
        """.

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
        """.

        The predict PyTorch ``DataLoader`` instance does not shuffle the
        dataset.

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
