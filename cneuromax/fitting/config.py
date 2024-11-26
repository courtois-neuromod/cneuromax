""":class:`.FittingRunConfig`."""

from dataclasses import dataclass
from typing import Annotated as An

from cneuromax.config import BaseRunConfig
from cneuromax.utils.beartype import one_of


@dataclass
class FittingRunConfig(BaseRunConfig):
    """Fitting ``run`` config.

    Args:
        device: Computing device to use for large matrix operations.
    """

    device: An[str, one_of("cpu", "gpu")] = "cpu"
