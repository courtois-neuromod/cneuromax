""":mod:`hydra-core` Deep Learning fitting config & validation."""

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING, DictConfig

from cneuromax.fitting.config import (
    BaseFittingHydraConfig,
    post_process_base_fitting_config,
    pre_process_base_fitting_config,
)


@dataclass(frozen=True)
class DeepLearningFittingHydraConfig(BaseFittingHydraConfig):
    """Structured :mod:`hydra-core` config for Deep Learning fitting.

    Args:
        trainer: Implicit (generated by :mod:`hydra-zen`)\
            ``TrainerHydraConfig`` instance that would have wrapped\
            :class:`lightning.pytorch.Trainer`.
        litmodule: Implicit (generated by :mod:`hydra-zen`)\
            ``LitModuleHydraConfig`` instance that would have wrapped\
            :class:`~.deeplearning.litmodule.BaseLitModule`.
        datamodule: Implicit (generated by :mod:`hydra-zen`)\
            ``DataModuleHydraConfig`` instance that would have wrapped\
            :class:`~.deeplearning.datamodule.BaseDataModule`.
        logger: Implicit (generated by :mod:`hydra-zen`)\
            ``LoggerHydraConfig`` instance that would have wrapped\
            :class:`lightning.pytorch.loggers.Logger`.
    """

    trainer: Any = MISSING
    litmodule: Any = MISSING
    datamodule: Any = MISSING
    logger: Any = MISSING


def pre_process_deep_learning_fitting_config(config: DictConfig) -> None:
    """Validates raw task config before it is made structured.

    Args:
        config: See :paramref:`~.pre_process_base_config.config`.

    """
    pre_process_base_fitting_config(config)


def post_process_deep_learning_fitting_config(
    config: DeepLearningFittingHydraConfig,
) -> None:
    """Validates the structured task config.

    Args:
        config: See :paramref:`~.post_process_base_config.config`.
    """
    post_process_base_fitting_config(config)
