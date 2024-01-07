""":class:`FittingSubtaskConfig`."""
from dataclasses import dataclass
from typing import Annotated as An

from cneuromax.config import BaseSubtaskConfig  # , BaseTaskConfig
from cneuromax.utils.beartype import one_of


@dataclass
class FittingSubtaskConfig(BaseSubtaskConfig):
    """Fitting ``subtask`` config.

    Args:
        device: Computing device to use for large matrix operations.
        copy_data_commands: List of commands to execute to transfer the\
            training data to the
            :paramref:`~.BaseSubtaskConfig.data_dir` directory.\
            This is useful when the training data is originally stored\
            in a different location than
            :paramref:`~.BaseSubtaskConfig.data_dir`.
    """

    device: An[str, one_of("cpu", "gpu")] = "cpu"
    copy_data_commands: list[str] | None = None
