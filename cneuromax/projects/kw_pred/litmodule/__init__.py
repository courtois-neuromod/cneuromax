r""":mod:`.kw_pred` :class:`lightning.pytorch.LightningModule``s."""

from hydra_zen import ZenStore

from cneuromax.utils.hydra_zen import fs_builds

from .dit import CustomDiT
from .kw_gen import KWGenerationLitModule, KWGenerationLitModuleConfig


def store_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` :mod:`.kw_pred` configs.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        fs_builds(
            KWGenerationLitModule,
            config=KWGenerationLitModuleConfig(),
        ),
        name="kw_gen",
        group="litmodule",
    )
    store(
        fs_builds(CustomDiT),
        name="custom_dit",
        group="litmodule/nnmodule",
    )
