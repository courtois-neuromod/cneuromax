"""Base DataModule class & related utilities."""

from abc import ABCMeta
from dataclasses import dataclass
from typing import Annotated as An
from typing import final

from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cneuromax.utils.annotations import ge, has_keys_annots, not_empty, one_of


@dataclass
class BaseDataset:
    """.

    Attributes:
        train: .
        val: .
        test: .
        predict: .
    """

    train: Dataset[Tensor] | None = None
    val: Dataset[Tensor] | None = None
    test: Dataset[Tensor] | None = None
    predict: Dataset[Tensor] | None = None


@dataclass
class BaseDataModuleConfig:
    """.

    Attributes:
        data_dir: .
        device: .
    """

    data_dir: An[str, not_empty()] = "${data_dir}"
    device: An[str, one_of("cpu", "gpu")] = "${device}"


class BaseDataModule(LightningDataModule, metaclass=ABCMeta):
    """Root Lightning ``DataModule`` class.

    With ``stage`` being any of ``"train"``, ``"val"``, ``"test"`` or
    ``"predict"``, subclasses need to properly define the
    ``dataset[stage]`` instance attribute(s) for each desired ``stage``.

    Attributes:
        config (``BaseDataModuleConfig``): .
        dataset (``BaseDataset``): .
        pin_memory (``bool``): Whether to copy tensors into device
            pinned memory before returning them (is set to ``True`` by
            default if using GPUs).
        per_device_batch_size (``int``): Per-device number of samples to
            load per iteration. Default value (``1``) is later
            overwritten with the use of a Lightning ``Tuner``.
        per_device_num_workers (``int``): Per-device number of CPU
            processes to use for data loading (``0`` means that the data
            will be loaded by each device's assigned CPU process).
            Default value (``0``) is later overwritten.
    """

    def __init__(self: "BaseDataModule", config: BaseDataModuleConfig) -> None:
        """Calls parent constructor & initializes instance attributes.

        Args:
            config: .
        """
        super().__init__()
        self.config = config
        self.dataset = BaseDataset()
        self.pin_memory = self.config.device == "gpu"
        self.per_device_batch_size = 1
        self.per_device_num_workers = 0

    @final
    def load_state_dict(
        self: "BaseDataModule",
        state_dict: An[
            dict[str, int],
            has_keys_annots(
                {
                    "per_device_batch_size": [int, ge(1)],
                    "per_device_num_workers": [int, ge(0)],
                },
            ),
        ],
    ) -> None:
        """Sets the instance's per-device batch_size & num_workers.

        Args:
            state_dict: .
        """
        self.per_device_batch_size = state_dict["per_device_batch_size"]
        self.per_device_num_workers = state_dict["per_device_num_workers"]

    @final
    def state_dict(self: "BaseDataModule") -> dict[str, int]:
        """.

        Returns:
            This instance's per-device batch size & number of workers
            inside a new dictionary.
        """
        return {
            "per_device_batch_size": self.per_device_batch_size,
            "per_device_num_workers": self.per_device_num_workers,
        }

    @final
    def x_dataloader(
        self: "BaseDataModule",
        dataset: Dataset[Tensor] | None,
        *,
        shuffle: bool = True,
    ) -> DataLoader[Tensor]:
        """Generic ``DataLoader`` factory method.

        Raises:
            AttributeError: If ``dataset`` is ``None``.

        Returns:
            A new PyTorch ``DataLoader`` instance.
        """
        if dataset is None:
            raise AttributeError

        return DataLoader(
            dataset=dataset,
            batch_size=self.per_device_batch_size,
            shuffle=shuffle,
            num_workers=self.per_device_num_workers,
            pin_memory=self.pin_memory,
        )

    @final
    def train_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls ``x_dataloader`` with train dataset.

        Returns:
            A new training PyTorch ``DataLoader`` instance.
        """
        return self.x_dataloader(dataset=self.dataset.train)

    @final
    def val_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls ``x_dataloader`` with val dataset.

        Returns:
            A new validation PyTorch ``DataLoader`` instance.
        """
        return self.x_dataloader(dataset=self.dataset.val)

    @final
    def test_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls ``x_dataloader`` with test dataset.

        Returns:
            A new testing PyTorch ``DataLoader`` instance.
        """
        return self.x_dataloader(dataset=self.dataset.test)

    @final
    def predict_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls ``x_dataloader`` with predict dataset.

        The predict PyTorch ``DataLoader`` instance does not shuffle the
        dataset.

        Returns:
            A new prediction PyTorch ``DataLoader`` instance.
        """
        return self.x_dataloader(dataset=self.dataset.test, shuffle=False)
