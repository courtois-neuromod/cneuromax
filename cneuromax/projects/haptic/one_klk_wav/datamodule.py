""":class:`OneKLKWavDataModule`."""
from typing import Annotated as An

from cneuromax.fitting.deeplearning.datamodule import BaseDataModule
from cneuromax.utils.beartype import one_of

from .dataset import OneKLKWavDataset


class OneKLKWavDataModule(BaseDataModule):
    """:mod:`one_klk_wav`` :class:`.BaseDataModule`."""

    def setup(
        self: "OneKLKWavDataModule",
        stage: An[str, one_of("fit", "validate", "test")],
    ) -> None:
        """Creates the train/val/test datasets.

        Args:
            stage: Current stage type.
        """
        if stage == "fit":
            self.datasets.train = OneKLKWavDataset(
                data_dir=self.config.data_dir,
            )
            self.datasets.val = OneKLKWavDataset(
                data_dir=self.config.data_dir,
            )
        else:  # stage == "test":
            error_msg = "No test dataset available."
            raise NotImplementedError(error_msg)
