""":mod:`lightning` DataModule + conf for MNIST classification task."""
from dataclasses import dataclass
from typing import Annotated as An

from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)
from cneuromax.utils.annotations import ge, lt, one_of


@dataclass
class MNISTClassificationDataModuleConfig(BaseDataModuleConfig):
    """Configuration for :class:`MNISTClassificationDataModule`.

    Args:
        val_percentage: Percentage of the training dataset to use for\
            validation.
    """

    val_percentage: An[float, ge(0), lt(1)] = 0.1


class MNISTClassificationDataModule(BaseDataModule):
    """MNIST Classification :mod:`lightning` DataModule.

    Args:
        config: The instance's configuration.

    Attributes:
        train_val_split (`tuple[float, float]`): The train/validation\
            split (sums to `1`).
        transform (:class:`~transforms.Compose`): The\
            :mod:`torchvision` dataset transformations.
    """

    def __init__(
        self: "MNISTClassificationDataModule",
        config: MNISTClassificationDataModuleConfig,
    ) -> None:
        super().__init__(config)
        self.train_val_split = (
            1 - config.val_percentage,
            config.val_percentage,
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ],
        )

    def prepare_data(self: "MNISTClassificationDataModule") -> None:
        """Downloads the MNIST dataset."""
        MNIST(root=self.config.data_dir, download=True)

    def setup(
        self: "MNISTClassificationDataModule",
        stage: An[str, one_of("fit", "test")],
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
