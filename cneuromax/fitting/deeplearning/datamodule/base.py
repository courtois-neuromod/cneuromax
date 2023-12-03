""":class:`BaseDataModule` & its helper dataclasses."""

from abc import ABCMeta
from dataclasses import dataclass
from typing import Annotated as An
from typing import final

from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cneuromax.utils.annotations import not_empty, one_of


@dataclass
class StageDataset:
    """Holds stage-specific :class:`~torch.utils.data.Dataset` objects.

    Args:
        train: Training dataset.
        val: Validation dataset.
        test: Testing dataset.
        predict: Prediction dataset.
    """

    train: Dataset[Tensor] | None = None
    val: Dataset[Tensor] | None = None
    test: Dataset[Tensor] | None = None
    predict: Dataset[Tensor] | None = None


@dataclass
class BaseDataModuleConfig:
    """Holds :class:`BaseDataModule` config values.

    Args:
        data_dir: See\
            :paramref:`~cneuromax.config.BaseHydraConfig.data_dir`.
        device: See\
            :paramref:`~cneuromax.fitting.config.BaseFittingHydraConfig.device`.
    """

    data_dir: An[str, not_empty()] = "${data_dir}"
    device: An[str, one_of("cpu", "gpu")] = "${device}"


class BaseDataModule(LightningDataModule, metaclass=ABCMeta):
    """Root :mod:`~lightning.pytorch.LightningDataModule` subclass.

    With ``stage`` being any of ``train``, ``val``, ``test`` or
    ``predict``, subclasses need to properly define the
    ``dataset.stage`` instance attribute(s) for each desired ``stage``.

    Args:
        config: See :class:`BaseDataModuleConfig`.

    Attributes:
        config (:class:`BaseDataModuleConfig`)
        dataset (:class:`StageDataset`)
        pin_memory (``bool``): Whether to copy tensors into device\
            pinned memory before returning them (is set to ``True`` by\
            default if\
            :paramref:`~cneuromax.fitting.config.BaseFittingHydraConfig.device`\
            is ``"gpu"``).
        per_device_batch_size (``int``): Per-device number of samples\
            to load per iteration. Default value (``1``) is later\
            overwritten through function\
            :func:`~cneuromax.fitting.deeplearning.fit.set_batch_size_and_num_workers`.
        per_device_num_workers (``int``): Per-device number of CPU\
            processes to use for data loading (``0`` means that the\
            data will be loaded by each device's assigned CPU\
            process). Default value (``0``) is later overwritten\
            through function\
            :func:`~cneuromax.fitting.deeplearning.fit.set_batch_size_and_num_workers`.
    """

    def __init__(self: "BaseDataModule", config: BaseDataModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.dataset = StageDataset()
        self.pin_memory = self.config.device == "gpu"
        self.per_device_batch_size = 1
        self.per_device_num_workers = 0

    @final
    def load_state_dict(
        self: "BaseDataModule",
        state_dict: dict[str, int],
    ) -> None:
        """Loads saved ``per_device_batch_size`` & ``num_workers`` vals.

        Args:
            state_dict: Dictionary containing values for\
                ``per_device_batch_size`` & ``num_workers``.
        """
        self.per_device_batch_size = state_dict["per_device_batch_size"]
        self.per_device_num_workers = state_dict["per_device_num_workers"]

    @final
    def state_dict(self: "BaseDataModule") -> dict[str, int]:
        """Returns ``per_device_batch_size`` & ``num_workers`` attribs.

        Returns:
            See :paramref:`~load_state_dict.state_dict`.
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
        """Generic :class:`~torch.utils.data.DataLoader` factory method.

        Args:
            dataset: The dataset to wrap with a\
                :class:`~torch.utils.data.DataLoader`
            shuffle: Whether to shuffle the dataset when iterating\
                over it.

        Raises:
            :class:`AttributeError`: If :paramref:`dataset` is ``None``.

        Returns:
            A new :class:`~torch.utils.data.DataLoader` instance\
                wrapping the :paramref:`dataset` argument.
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
        """Calls :meth:`x_dataloader` with ``dataset.train`` attribute.

        Returns:
            A new training :class:`torch.utils.data.DataLoader`\
                instance.
        """
        return self.x_dataloader(dataset=self.dataset.train)

    @final
    def val_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls :meth:`x_dataloader` with ``dataset.val`` attribute.

        Returns:
            A new validation :class:`~torch.utils.data.DataLoader`\
                instance.
        """
        return self.x_dataloader(dataset=self.dataset.val)

    @final
    def test_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls :meth:`x_dataloader` with ``dataset.test`` attribute.

        Returns:
            A new testing :class:`~torch.utils.data.DataLoader`\
                instance.
        """
        return self.x_dataloader(dataset=self.dataset.test)

    @final
    def predict_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls :meth:`x_dataloader` w/ ``dataset.predict`` attribute.

        Returns:
            A new prediction :class:`~torch.utils.data.DataLoader`\
                instance that does not shuffle the dataset.
        """
        return self.x_dataloader(dataset=self.dataset.predict, shuffle=False)
