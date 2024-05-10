""":class:`KWPredDataModule` + its config."""

from dataclasses import dataclass
from typing import Annotated as An

from torch.utils.data import Subset

from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)
from cneuromax.utils.beartype import ge, lt, one_of

from .dataset import KWPredDataset


@dataclass
class KWPredDatamoduleConfig(BaseDataModuleConfig):
    """Holds :class:`KWPredDataModule` config values.

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
        dataset (:class:`KWPredDataset`): See :paramref:`dataset`.
        train_val_split (``tuple[float, float]``): Percentages of the\
            data to use for training and validation, respectively.\
            Sums to 1.
    """

    def __init__(
        self: "KWPredDataModule",
        config: KWPredDatamoduleConfig,
        dataset: KWPredDataset,
    ) -> None:
        super().__init__(config=config)
        self.config: KWPredDatamoduleConfig
        self.dataset = dataset
        self.train_val_split = (
            1 - config.val_percentage,
            config.val_percentage,
        )

    def setup(
        self: "KWPredDataModule",
        stage: An[str, one_of("fit", "validate", "test")],
    ) -> None:
        """Creates the train/val/test datasets.

        Args:
            stage: Current stage type.
        """
        if stage == "fit":
            last_train_idx = int(
                len(self.dataset) * (1 - self.config.val_percentage),
            )
            self.datasets.train = Subset(
                self.dataset,
                indices=range(last_train_idx),
            )
            self.datasets.val = Subset(
                self.dataset,
                indices=range(last_train_idx, len(self.dataset)),
            )
        else:  # stage == "test":
            try:
                self.datasets.test = self.datasets.val
            except AttributeError as e:
                error_msg = "No test dataset available."
                raise AttributeError(error_msg) from e
