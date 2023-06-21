"""."""

from typing import TYPE_CHECKING

from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from cneuroml.deeplearning.common.datamodule import (
    BaseDataModule,
)

if TYPE_CHECKING:
    from cneuroml.deeplearning.common.datamodule import BaseDataModuleConfig


class MNISTDataModule(BaseDataModule):
    """.

    Attributes:
        config (``BaseDataModuleConfig``): .
        dataset (``dict[Literal["train", "val", "test", "predict"],
            Dataset]``): .
        train_val_split (``list[float, float]``): The train/validation
            split (sums to 1).
        transforms (``torchvision.transforms.Compose``): The
            transformation(s) to apply to the dataset.
    """

    def __init__(
        self: "MNISTDataModule",
        config: "BaseDataModuleConfig",
        val_percentage: float = 0.1,
        transforms: transforms.Compose = transforms.ToTensor,
    ) -> None:
        """Calls parent constructor and stores arguments.

        Args:
            config: .
            val_percentage: Percentage of the training dataset to use
                for validation.
            transforms: The transformation(s) to apply to the dataset.
        """
        super().__init__(config)
        self.train_val_split = [1 - val_percentage, val_percentage]
        self.transforms = transforms

    def prepare_data(self: "MNISTDataModule") -> None:
        """Downloads the MNIST dataset."""
        MNIST(self.config.data_dir, train=True, download=True)
        MNIST(self.config.data_dir, train=False, download=True)

    def setup(self: "MNISTDataModule", stage: str) -> None:
        """Creates the train/val/test/predict datasets.

        Args:
            stage: ``"fit"``, ``"test"``, or ``"predict"``.
        """
        if stage == "fit":
            mnist_full = MNIST(
                self.config.data_dir,
                train=True,
                transform=self.transforms,
            )
            self.dataset["train"], self.dataset["val"] = random_split(
                mnist_full,
                self.train_val_split,
            )

        if stage == "test":
            self.dataset["test"] = MNIST(
                self.config.data_dir,
                train=False,
                transform=self.transforms,
            )

        if stage == "predict":
            self.dataset["predict"] = MNIST(
                self.config.data_dir,
                train=False,
                transform=self.transforms,
            )
