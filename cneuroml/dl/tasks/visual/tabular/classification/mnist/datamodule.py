"""MNIST Classification DataModule."""

from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from cneuroml.dl.base import BaseDataModule, BaseDataModuleConfig


class MNISTClassificationDataModule(BaseDataModule):
    """DataModule for MNIST Classification.

    Attributes:
        config (``BaseDataModuleConfig``): The base dataclass
            configuration instance.
        dataset (``dict[Literal["train", "val", "test", "predict"],
            Dataset]``): The dataset dictionary containing the PyTorch
            ``Dataset`` instance(s) for each desired stage.
        train_val_split (``list[float, float]``): The train/validation
            split.
        transform (``torchvision.transforms.Compose``): The
            transformation(s) to apply to the dataset.
    """

    def __init__(
        self: "MNISTClassificationDataModule",
        config: "BaseDataModuleConfig",
        val_percentage: float = 0.1,
        transforms: transforms.Compose = transforms.ToTensor,
    ) -> None:
        """Constructor.

        Calls parent constructor and stores arguments.

        Args:
            config: A ``BaseDataModuleConfig`` instance.
            val_percentage: Percentage of the training dataset to use
                for validation.
            transforms: A ``torchvision.transforms.Compose`` instance.
        """
        super().__init__(config)
        self.train_val_split = [1 - val_percentage, val_percentage]
        self.transform = transforms

    def prepare_data(self: "MNISTClassificationDataModule") -> None:
        """Downloads the MNIST dataset."""
        MNIST(self.config.data_dir, train=True, download=True)
        MNIST(self.config.data_dir, train=False, download=True)

    def setup(self: "MNISTClassificationDataModule", stage: str) -> None:
        """Creates the train/val/test/predict datasets.

        Args:
            stage: ``"fit"``, ``"test"``, or ``"predict"``.
        """
        if stage == "fit":
            mnist_full = MNIST(
                self.config.data_dir,
                train=True,
                transform=self.transform,
            )
            self.dataset["train"], self.dataset["val"] = random_split(
                mnist_full,
                self.train_val_split,
            )

        if stage == "test":
            self.dataset["test"] = MNIST(
                self.config.data_dir,
                train=False,
                transform=self.transform,
            )

        if stage == "predict":
            self.dataset["predict"] = MNIST(
                self.config.data_dir,
                train=False,
                transform=self.transform,
            )
