""":mod:`hydra-core` DL + NE fitting config & validation."""

from dataclasses import dataclass

from omegaconf import DictConfig

from cneuromax.fitting.config import (
    BaseFittingHydraConfig,
    post_process_base_fitting_config,
    pre_process_base_fitting_config,
)


@dataclass(frozen=True)
class DeepLNeuroEvoFittingHydraConfig(BaseFittingHydraConfig):
    """Structured :mod:`hydra-core` config for DL + NE fitting."""


def pre_process_deeplneuroevo_fitting_config(config: DictConfig) -> None:
    """Validates :paramref:`config` before it is made structured.

    Args:
        config: See :paramref:`~.pre_process_base_config.config`.
    """
    pre_process_base_fitting_config(config)


def post_process_deeplneuroevo_fitting_config(
    config: DeepLNeuroEvoFittingHydraConfig,
) -> None:
    """Validates the structured task config.

    Args:
        config: See :paramref:`~.post_process_base_config.config`.
    """
    post_process_base_fitting_config(config)
