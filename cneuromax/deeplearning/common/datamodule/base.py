"""."""

from abc import ABCMeta
from dataclasses import dataclass
from typing import Annotated, final

from beartype.vale import Is
from hydra_zen import store
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


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


@store(name="base", group="datamodule/config")
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
    """

    data_dir: str = "${data_dir}"
    per_device_batch_size: Annotated[int, Is[lambda x: x > 0]] = 1
    per_device_num_workers: Annotated[int, Is[lambda x: x >= 0]] = 0
    device_type: Annotated[str, Is[lambda x: x in ("cpu", "gpu")]] = "gpu"


@store(name="base", group="datamodule")
class BaseDataModule(LightningDataModule, metaclass=ABCMeta):
    """.

    With ``stage`` being any of ``"train"``, ``"val"``, ``"test"`` or
    ``"predict"``, children of this class need to properly define
    ``dataset[stage]`` instance attribute(s) for each desired ``stage``.

    Attributes:
        config (``BaseDataModuleConfig``): .
        dataset (``dict[Literal["train", "val", "test", "predict"],\
        Dataset]``): .
        pin_memory (``bool``): Whether to copy tensors into device
            pinned memory before returning them (is set to ``True`` by
            default if using GPUs).
    """

    def __init__(self: "BaseDataModule", config: BaseDataModuleConfig) -> None:
        """Calls parent constructor.

        Args:
            config: .
        """
        super().__init__()
        self.config: BaseDataModuleConfig = config
        self.pin_memory: bool = self.config.device_type == "gpu"
        self.dataset: BaseDataset = BaseDataset()

    @final
    def load_state_dict(
        self: "BaseDataModule",
        state_dict: dict[
            Annotated[
                str,
                Is[lambda x: x in ("per_device_batch_size")],
            ],
            Annotated[int, Is[lambda x: x > 0]],
        ],
    ) -> None:
        """Sets the instance's batch size from a dictionary value.

        Args:
            state_dict: .
        """
        self.config.per_device_batch_size = state_dict["per_device_batch_size"]

    @final
    def state_dict(
        self: "BaseDataModule",
    ) -> dict[
        Annotated[
            str,
            Is[lambda x: x in ("per_device_batch_size")],
        ],
        Annotated[int, Is[lambda x: x > 0]],
    ]:
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
            AttributeError: If the instance's ``dataset.train``
                attribute is ``None``.
        """
        if self.dataset.train is None:
            raise AttributeError

        return DataLoader(
            dataset=self.dataset.train,
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
            AttributeError: If the instance's ``dataset.val`` attribute
                is ``None``.
        """
        if self.dataset.val is None:
            raise AttributeError

        return DataLoader(
            dataset=self.dataset.val,
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
            AttributeError: If the instance's ``dataset.test``
                attribute is ``None``.
        """
        if self.dataset.test is None:
            raise AttributeError

        return DataLoader(
            dataset=self.dataset.test,
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
            AttributeError: If the instance's ``dataset.predict``
                attribute is ``None``.
        """
        if self.dataset.predict is None:
            raise AttributeError

        return DataLoader(
            dataset=self.dataset.predict,
            batch_size=self.config.per_device_batch_size,
            shuffle=False,
            num_workers=self.config.per_device_num_workers,
            pin_memory=self.config.pin_memory,
        )
