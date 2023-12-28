""":mod:`hydra-core` DL + NE fitting config & validation."""

from dataclasses import dataclass

from omegaconf import DictConfig

from cneuromax.fitting.config import (
    BaseFittingHydraConfig,
    post_process_base_fitting_config,
    pre_process_base_fitting_config,
)


@dataclass(frozen=True)
class HybridFittingHydraConfig(BaseFittingHydraConfig):
    """Structured :mod:`hydra-core` config for DL + NE fitting."""


def pre_process_hybrid_fitting_config(config: DictConfig) -> None:
    """Validates raw task config before it is made structured.

    Args:
        config: The raw task config.

    """
    pre_process_base_fitting_config(config)


def post_process_hybrid_fitting_config(
    config: HybridFittingHydraConfig,
) -> None:
    """Post-processes the :mod:`hydra-core` config after it is resolved.

    Args:
        config: The processed :mod:`hydra-core` config.
    """
    post_process_base_fitting_config(config)
