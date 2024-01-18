""":class:`ARDataModule` + its config."""
from dataclasses import dataclass
from typing import Annotated as An

from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)
from cneuromax.utils.beartype import one_of


@dataclass
class ARDataModuleConfig(BaseDataModuleConfig):
    """Configuration for :class:`ARDataModule`."""


class ARDataModule(BaseDataModule):
    """Autoregression :mod:`lightning` DataModule.

    Args:
        config: See :class:`ARDataModuleConfig`.
    """

    def __init__(
        self: "ARDataModule",
        config: ARDataModuleConfig,
    ) -> None:
        super().__init__(config=config)

    def setup(
        self: "ARDataModule",
        stage: An[str, one_of("fit", "validate", "test")],
    ) -> None:
        """Creates the train/val/test datasets.

        Args:
            stage: Current stage type.
        """
        if stage == "fit":
            pass

        else:  # stage == "test":
            pass
