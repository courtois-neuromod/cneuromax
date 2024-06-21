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
class DeepLearningSubtaskConfig(FittingSubtaskConfig):
    """Deep Learning ``subtask`` config.

    Args:
        compile: Whether to compile the :class:`.BaseLitModule`\
            before training. Requires\
            :paramref:`FittingSubtaskConfig.device` to be set to\
            ``"gpu"`` & a CUDA 7+ compatible GPU.
        save_every_n_train_steps: The frequency at which to save\
            training checkpoints.
        ckpt_path: The path to a checkpoint to resume training from.
    """

    compile: bool = False
    save_every_n_train_steps: int | None = 1
    ckpt_path: str = "last"


@dataclass
class DeepLearningTaskConfig(
    make_config(  # type: ignore[misc]
        trainer=pfs_builds(Trainer),
        datamodule=fs_builds(BaseDataModule, config=BaseDataModuleConfig()),
        litmodule=fs_builds(BaseLitModule),
        logger=pfs_builds(WandbLogger),
        config=fs_builds(DeepLearningSubtaskConfig),
    ),
):
    """Deep Learning ``task`` config.

    Args:
        defaults: Hydra defaults.
        trainer: See :class:`~lightning.pytorch.Trainer`.
        datamodule: See :class:`.BaseDataModule`.
        litmodule: See :class:`.BaseLitModule`.
        logger: See\
            :class:`~lightning.pytorch.loggers.wandb.WandbLogger`.
        config: See :class:`DeepLearningSubtaskConfig`.
    """

    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"trainer": "base"},
            {"litmodule/nnmodule": "mlp"},
            {"litmodule/scheduler": "constant"},
            {"litmodule/optimizer": "adamw"},
            {"logger": "wandb_simexp"},
            "project",
            "task",
            {"task": None},
        ],
    )
