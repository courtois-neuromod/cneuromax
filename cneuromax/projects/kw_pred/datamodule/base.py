""":class:`KWPredDataset` + :class:`KWPredDataModule` & its config, ."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated as An
from functools import partial
from torch import Tensor
from torch.utils.data import Dataset, random_split

from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)
from .dataset import KWPredDataset
from cneuromax.utils.beartype import ge, lt, one_of


@dataclass
class KWPredDatamoduleConfig(BaseDataModuleConfig):
    """:class:`KWPredDataModule` config.

    Args:
        val_percentage: Percentage of the training dataset to use for\
            validation.
    """

    val_percentage: An[float, ge(0), lt(1)] = 0.1


class KWPredDataModule(BaseDataModule):
    """:mod:`.kw_pred` :class:`.BaseDataModule`.

    Args:
        config: See :class:`KWPredDatamoduleConfig`.
        dataset: See :class:`KWPredDataset`.

    Attributes:
        config (:class:`KWPredDatamoduleConfig`): See\
            :paramref:`config`.
        train_val_split (``tuple[float, float]``): Percentages of the\
            data to use for training and validation, respectively.\
            Sums to 1.
        paths (:class:`.KWPredPaths`): See :class:`.KWPredPaths`.
    """

    def __init__(
        self: "KWPredDataModule",
        config: KWPredDatamoduleConfig,
        dataset: partial[KWPredDataset],
    ) -> None:
        super().__init__(config=config)
        self.config: KWPredDatamoduleConfig
        self.train_val_split = (
            1 - config.val_percentage,
            config.val_percentage,
        )
        self.dataset_partial = dataset


    def setup(
        self: "KWPredDataModule",
        stage: An[str, one_of("fit", "validate", "test")],
    ) -> None:
        """Creates the train/val/test datasets.

        Args:
            stage: Current stage type.
        """
        if stage == "fit":
            dataset = KWPredDataset(paths=self.paths,
            self.datasets.train, self.datasets.val = random_split(
                dataset=dataset,
                lengths=self.train_val_split,
            )
        else:  # stage == "test":
            error_msg = "No test dataset available."
            raise NotImplementedError(error_msg)
