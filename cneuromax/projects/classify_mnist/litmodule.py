""":class:`.MNISTClassificationLitModule`."""

from typing import Any

import wandb

from cneuromax.fitting.deeplearning.litmodule.classification import (
    BaseClassificationLitModule,
)


class MNISTClassificationLitModule(BaseClassificationLitModule):
    """``project`` :class:`BaseClassificationLitModule`."""

    def __init__(
        self: "MNISTClassificationLitModule",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)
        self.to_wandb_media = wandb.Image
