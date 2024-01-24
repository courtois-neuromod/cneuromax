""":class:`KLKWavDataModule` & its config."""
from dataclasses import dataclass
from typing import Annotated as An

from torch.utils.data import random_split

from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)
from cneuromax.utils.beartype import ge, lt, one_of

from .dataset import KLKWavDataset


@dataclass
class KLKWavDataModuleConfig(BaseDataModuleConfig):
    """Holds :class:`KLKWavDataModule` config values.

    Args:
        val_percentage: Percentage of the training dataset to use for\
            validation.
    """

    val_percentage: An[float, ge(0), lt(1)] = 0.1


class KLKWavDataModule(BaseDataModule):
    """:mod:`klk_wav`` :class:`.BaseDataModule`.

    Args:
        config: See :class:`KLKWavDataModuleConfig`.
    """

    def __init__(
        self: "KLKWavDataModule",
        config: KLKWavDataModuleConfig,
    ) -> None:
        super().__init__(config=config)
        self.train_val_split = (
            1 - config.val_percentage,
            config.val_percentage,
        )

    def setup(
        self: "KLKWavDataModule",
        stage: An[str, one_of("fit", "validate", "test")],
    ) -> None:
        """Creates the train/val/test datasets.

        Args:
            stage: Current stage type.
        """
        if stage == "fit":
            dataset = KLKWavDataset(data_dir=self.config.data_dir)
            self.datasets.train, self.datasets.val = random_split(
                dataset=dataset,
                lengths=self.train_val_split,
            )
        else:  # stage == "test":
            error_msg = "No test dataset available."
            raise NotImplementedError(error_msg)
