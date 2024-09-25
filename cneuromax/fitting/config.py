""":class:`.FittingSubtaskConfig`."""

from dataclasses import dataclass
from typing import Annotated as An

from cneuromax.config import BaseSubtaskConfig
from cneuromax.utils.beartype import one_of


@dataclass
class FittingSubtaskConfig(BaseSubtaskConfig):
    """Fitting ``subtask`` config.

    Args:
        device: Computing device to use for large matrix operations.
    """

    device: An[str, one_of("cpu", "gpu")] = "cpu"
