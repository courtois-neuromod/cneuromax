""":class:`.MNISTDataModule` & its config."""

from dataclasses import dataclass
from typing import Annotated as An

from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)
from cneuromax.utils.beartype import ge, lt, one_of


@dataclass
class MNISTDataModuleConfig(BaseDataModuleConfig):
    """Holds :class:`MNISTDataModule` config values.

    Args:
        val_percentage: Percentage of the training dataset to use for
            validation.
    """

    val_percentage: An[float, ge(0), lt(1)] = 0.005


class MNISTDataModule(BaseDataModule):
    """``project`` :class:`.BaseDataModule`.

    Args:
        config: See :class:`MNISTDataModuleConfig`.

    Attributes:
        train_val_split (`tuple[float, float]`): The train/validation
            split (sums to `1`).
        transform (:class:`~transforms.Compose`): The
            :mod:`torchvision` dataset transformations.
    """

    def __init__(
        self: "MNISTDataModule",
        config: MNISTDataModuleConfig,
    ) -> None:
        super().__init__(config=config)
        self.train_val_split = (
            1 - config.val_percentage,
            config.val_percentage,
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Pre-computer mean and std for the MNIST dataset.
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ],
        )

    def prepare_data(self: "MNISTDataModule") -> None:
        """Downloads the MNIST dataset."""
        MNIST(root=self.config.data_dir, download=True)

    def setup(
        self: "MNISTDataModule",
        stage: An[str, one_of("fit", "validate", "test")],
    ) -> None:
        """Creates the train/val/test datasets.

        Args:
            stage: Current stage type.
        """
        if stage == "fit":
            mnist_full = MNIST(
                root=self.config.data_dir,
                train=True,
                transform=self.transform,
            )
            self.datasets.train, self.datasets.val = random_split(
                dataset=mnist_full,
                lengths=self.train_val_split,
            )

        else:  # stage == "test":
            self.datasets.test = MNIST(
                root=self.config.data_dir,
                train=False,
                transform=self.transform,
            )
