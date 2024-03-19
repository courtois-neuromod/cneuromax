""":class:`PEFTLitModule`."""

from abc import ABC
from dataclasses import dataclass
from typing import Any

from peft.config import PeftConfig
from peft.mapping import get_peft_model

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
)


@dataclass
class PEFTLitModule(BaseLitModule, ABC):
    """``project`` :class:`~PEFTLitModule`."""

    def __init__(
        self: "PEFTLitModule",
        peft_config: PeftConfig,
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> None:
        """."""
        super().__init__(
            *args,
            **kwargs,
        )
        self.nnmodule = get_peft_model(self.nnmodule, peft_config)
