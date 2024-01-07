""":class:`DeepLearningTaskConfig`."""
from dataclasses import dataclass, field
from typing import Any

from hydra_zen import make_config
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from cneuromax.fitting.config import (
    FittingSubtaskConfig,
)
from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)
from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.utils.hydra_zen import (
    fs_builds,
    pfs_builds,
)


@dataclass
class DeepLearningTaskConfig(
    make_config(  # type: ignore[misc]
        trainer=pfs_builds(Trainer),
        logger=pfs_builds(WandbLogger),
        datamodule=fs_builds(BaseDataModule, config=BaseDataModuleConfig()),
        litmodule=fs_builds(BaseLitModule),
        config=fs_builds(FittingSubtaskConfig),
    ),
):
    """Deep Learning ``task`` config.

    Args:
        defaults: Hydra defaults.
        trainer: See :class:`~lightning.pytorch.Trainer`.
        logger: See\
            :class:`~lightning.pytorch.loggers.wandb.WandbLogger`.
        datamodule: See :class:`.BaseDataModule`.
        litmodule: See :class:`.BaseLitModule`.
        config: See :class:`.FittingSubtaskConfig`.
    """

    defaults: list[Any] = field(
        default_factory=lambda: [
            {"hydra/launcher": "submitit_local"},
            {"trainer": "base"},
            {"litmodule/nnmodule": "mlp"},
            {"litmodule/scheduler": "constant"},
            {"litmodule/optimizer": "adamw"},
            {"logger": "wandb"},
            "_self_",
            {"task": None},
        ],
    )
