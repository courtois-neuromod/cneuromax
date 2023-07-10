"""."""

from dataclasses import dataclass

from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from cneuromax.common.utils.annotations import (
    float_is_ge0_le1,
    str_is_fit_or_test,
)
from cneuromax.deeplearning.common.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)


@dataclass
class MNISTClassificationDataModuleConfig(BaseDataModuleConfig):
    """."""

    val_percentage: float_is_ge0_le1
    fit_dataset_mean: tuple[float] = (0.1307,)
    fit_dataset_std: tuple[float] = (0.3081,)


class MNISTClassificationDataModule(BaseDataModule):
    """.

    Attributes:
        train_val_split (``tuple[float, float]``): The train/validation
            split (sums to ``1``).
        transform (``transforms.Compose``): The ``torchvision`` dataset
            transformations.
    """

    def __init__(
        self: "MNISTClassificationDataModule",
        config: MNISTClassificationDataModuleConfig,
    ) -> None:
        """.

        Calls parent constructor, type-hints the config, sets the
        train/val split and creates the dataset transform.

        Args:
            config: .
        """
        super().__init__(config)
        self.config: MNISTClassificationDataModuleConfig
        self.train_val_split: tuple[float, float] = (
            1 - self.config.val_percentage,
            self.config.val_percentage,
        )
        self.transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.fit_dataset_mean,
                    std=self.config.fit_dataset_std,
                ),
            ],
        )

    def prepare_data(self: "MNISTClassificationDataModule") -> None:
        """Downloads the MNIST dataset."""
        MNIST(root=self.config.data_dir, download=True)

    def setup(
        self: "MNISTClassificationDataModule",
        stage: str_is_fit_or_test,
    ) -> None:
        """Creates the train/val/test datasets.

        Args:
            stage: .
        """
        if stage == "fit":
            mnist_full: MNIST = MNIST(
                root=self.config.data_dir,
                train=True,
                transform=self.transform,
            )
            self.dataset.train, self.dataset.val = random_split(
                dataset=mnist_full,
                lengths=self.train_val_split,
            )

        else:  # stage == "test":
            self.dataset.test = MNIST(
                root=self.config.data_dir,
                train=False,
                transform=self.transform,
            )
