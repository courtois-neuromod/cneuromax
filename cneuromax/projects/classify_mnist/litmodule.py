""":class:`.MNISTClassificationLitModule`."""

import wandb

from cneuromax.fitting.deeplearning.litmodule.classification import (
    BaseClassificationLitModule,
)


class MNISTClassificationLitModule(BaseClassificationLitModule):
    """``project`` :class:`BaseClassificationLitModule`."""

    @property
    def wandb_media_x(self):  # type: ignore[no-untyped-def] # noqa: ANN201, D102
        return wandb.Image
