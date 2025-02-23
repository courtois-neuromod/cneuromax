""":class:`.BaseDataModule` + its datasets/config classes."""

from abc import ABC
from dataclasses import dataclass
from typing import Annotated as An
from typing import final

from datasets import Dataset as HFDataset
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cneuromax.utils.beartype import ge, not_empty, one_of


@dataclass
class Datasets:
    """Holds phase-specific :class:`torch.utils.data.Dataset` objects.

    Using the word ``phase`` to not overload
    `Lightning <https://lightning.ai/>`_ ``stage``
    terminology used for ``fit``, ``validate`` and ``test``.

    Args:
        train
        val
        test
        predict
    """

    train: Dataset[Tensor | dict[str, Tensor]] | HFDataset | None = None
    val: Dataset[Tensor | dict[str, Tensor]] | HFDataset | None = None
    test: Dataset[Tensor | dict[str, Tensor]] | HFDataset | None = None
    predict: Dataset[Tensor | dict[str, Tensor]] | HFDataset | None = None


@dataclass
class BaseDataModuleConfig:
    """Holds :class:`BaseDataModule` config values.

    Args:
        data_dir: See :paramref:`~.BaseSubtaskConfig.data_dir`.
        device: See :paramref:`~.FittingSubtaskConfig.device`.
        max_per_device_batch_size: See
            :attr:`~BaseDataModule.per_device_batch_size`. Sets an
            upper bound on the aforementioned attribute.
        fixed_per_device_batch_size: See
            :attr:`~BaseDataModule.per_device_batch_size`. Setting this
            value skips the batch size search in
            :func:`.find_good_per_device_batch_size` which is
            not recommended for resource efficiency.
        fixed_per_device_num_workers: See
            :attr:`~BaseDataModule.per_device_num_workers`. Setting
            this value skips the num workers search in
            :func:`.find_good_per_device_num_workers` which is
            not recommended for resource efficiency.
        shuffle_train_dataset
        shuffle_val_dataset
    """

    data_dir: An[str, not_empty()] = "${config.data_dir}"
    device: An[str, one_of("cpu", "gpu")] = "${config.device}"
    max_per_device_batch_size: An[int, ge(1)] | None = None
    fixed_per_device_batch_size: An[int, ge(1)] | None = None
    fixed_per_device_num_workers: An[int, ge(0)] | None = None
    shuffle_train_dataset: bool = True
    shuffle_val_dataset: bool = True


class BaseDataModule(LightningDataModule, ABC):
    """Base :class:`lightning.pytorch.core.LightningDataModule`.

    With ``<phase>`` being any of ``train``, ``val``, ``test`` or
    ``predict``, subclasses need to properly define the
    ``datasets.<phase>`` attribute(s) for each desired phase.

    Args:
        config

    Attributes:
        config (BaseDataModuleConfig)
        datasets (Datasets)
        collate_fn (typing.Callable): See ``collate_fn`` argument in
            :class:`torch.utils.data.DataLoader`.
        pin_memory (bool): Whether to copy tensors into device
            pinned memory before returning them (is set to ``True`` by
            default if :paramref:`~BaseDataModuleConfig.device` is
            ``"gpu"``).
        per_device_batch_size (int): Per-device number of samples
            to load per iteration. Temporary value (``1``) is
            overwritten in :func:`.set_batch_size_and_num_workers`.
        per_device_num_workers (int): Per-device number of CPU
            processes to use for data loading (``0`` means that the
            data will be loaded by each device's assigned CPU
            process). Temporary value (``0``) is later overwritten
            in :func:`.set_batch_size_and_num_workers`.
    """

    def __init__(self: "BaseDataModule", config: BaseDataModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.datasets = Datasets()
        self.collate_fn = None
        self.pin_memory = self.config.device == "gpu"
        self.per_device_batch_size = 1
        self.per_device_num_workers = 0

    @final
    def load_state_dict(
        self: "BaseDataModule",
        state_dict: dict[str, int],
    ) -> None:
        """Replace instance attrib vals w/ :paramref:`state_dict` vals.

        Args:
            state_dict: Dictionary containing values to override
                :attr:`per_device_batch_size` &
                :attr:`per_device_num_workers`.
        """
        self.per_device_batch_size = state_dict["per_device_batch_size"]
        self.per_device_num_workers = state_dict["per_device_num_workers"]

    @final
    def state_dict(self: "BaseDataModule") -> dict[str, int]:
        """Returns instance attribute values.

        Returns:
            A new dictionary containing attribute values
                :attr:`per_device_batch_size` &
                :attr:`per_device_num_workers`.
        """
        return {
            "per_device_batch_size": self.per_device_batch_size,
            "per_device_num_workers": self.per_device_num_workers,
        }

    @final
    def x_dataloader(
        self: "BaseDataModule",
        dataset: Dataset[Tensor] | HFDataset | None,
        *,
        shuffle: bool = True,
    ) -> DataLoader[Tensor]:
        """Generic :class:`torch.utils.data.DataLoader` factory method.

        Args:
            dataset: A :mod:`torch` ``Dataset`` to wrap with a
                :class:`torch.utils.data.DataLoader`
            shuffle: Whether to shuffle the dataset when iterating
                over it.

        Raises:
            AttributeError: If :paramref:`dataset` is :obj:`None`.

        Returns:
            A new
                :class:`torch.utils.data.DataLoader` instance
                wrapping the :paramref:`dataset` argument.
        """
        if dataset is None:
            raise AttributeError
        return DataLoader(
            dataset=dataset,
            batch_size=self.per_device_batch_size,
            shuffle=shuffle,
            num_workers=self.per_device_num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @final
    def train_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls :meth:`x_dataloader` w/ :attr:`datasets` ``.train``.

        Returns:
            A new training
                :class:`torch.utils.data.DataLoader` instance.
        """
        return self.x_dataloader(
            dataset=self.datasets.train,
            shuffle=self.config.shuffle_train_dataset,
        )

    def val_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls :meth:`x_dataloader` w/ :attr:`datasets` ``.val``.

        Returns:
            A new validation
                :class:`torch.utils.data.DataLoader` instance.
        """
        return self.x_dataloader(
            dataset=self.datasets.val,
            shuffle=self.config.shuffle_val_dataset,
        )

    @final
    def test_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls :meth:`x_dataloader` w/ :attr:`datasets` ``.test``.

        Returns:
            A new testing
                :class:`torch.utils.data.DataLoader` instance.
        """
        return self.x_dataloader(dataset=self.datasets.test, shuffle=False)

    @final
    def predict_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        """Calls :meth:`x_dataloader` w/ :attr:`datasets` ``.predict``.

        Returns:
            A new prediction
                :class:`torch.utils.data.DataLoader` instance that does
                not shuffle the dataset.
        """
        return self.x_dataloader(dataset=self.datasets.predict, shuffle=False)
