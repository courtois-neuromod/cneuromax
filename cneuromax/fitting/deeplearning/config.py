""":mod:`hydra-core` Deep Learning fitting config & validation."""

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING, DictConfig

from cneuromax.fitting.config import (
    BaseFittingHydraConfig,
    post_process_base_fitting_config,
    pre_process_base_fitting_config,
)


@dataclass(frozen=)
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
    """Pre-processes config from :func:`hydra.main` before resolution.

    Args:
        config: The not yet processed :mod:`hydra-core` config.

    """
    pre_process_base_fitting_config(config)


def post_process_deep_learning_fitting_config(
    config: BaseFittingHydraConfig,
) -> None:
    """Post-processes the :mod:`hydra-core` config after it is resolved.

    Args:
        config: The processed :mod:`hydra-core` config.
    """
    post_process_base_fitting_config(config)
