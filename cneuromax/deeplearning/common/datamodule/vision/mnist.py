"""."""

from dataclasses import dataclass
from typing import Annotated

from beartype.vale import Is
from hydra_zen import store
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from cneuromax.deeplearning.common.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)


@store(name="mnist", group="datamodule/config")
@dataclass
class MNISTDataModuleConfig(BaseDataModuleConfig):
    """.

    Attributes:
        val_percentage: Percentage of the training dataset to use
            for validation (float in ``]0, 1[``).
        fit_dataset_mean: .
        fit_dataset_std: .
    """

    val_percentage: Annotated[float, Is[lambda x: 0 < x < 1]] = 0.1
    fit_dataset_mean: tuple[float] = (0.1307,)
    fit_dataset_std: tuple[float] = (0.3081,)


@store(name="mnist", group="datamodule")
class MNISTDataModule(BaseDataModule):
    """.

    Attributes:
        train_val_split (``tuple[float, float]``): The train/validation
            split (sums to ``1``).
        transform (``transforms.Compose``): The ``torchvision`` dataset
            transformations.
    """

    def __init__(
        self: "MNISTDataModule",
        config: MNISTDataModuleConfig,
    ) -> None:
        """.

        Calls parent constructor, type-hints the config, sets the
        train/validation split and creates the dataset transform.

        Args:
            config: .
        """
        super().__init__(config)
        self.config: MNISTDataModuleConfig
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

    def prepare_data(self: "MNISTDataModule") -> None:
        """Downloads the MNIST dataset."""
        MNIST(root=self.config.data_dir, download=True)

    def setup(
        self: "MNISTDataModule",
        stage: Annotated[str, Is[lambda x: x in ["fit", "test"]]],
    ) -> None:
        """Creates the train/val/test datasets.

        Args:
            stage: ``"fit"`` or ``"test"``.
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
