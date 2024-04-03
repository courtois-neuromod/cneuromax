r""":mod:`.kw_pred` :class:`lightning.pytorch.LightningModule``s."""

from denoising_diffusion_pytorch import Unet1D
from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.litmodule import BaseLitModuleConfig
from cneuromax.utils.hydra_zen import fs_builds

from .dit import CustomDiT
from .kw_gen import KWGenerationLitModule, KWGenerationLitModuleConfig
from .unc_kw_gen import UnconditionalKWGenerationLitModule


def store_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` :mod:`.kw_pred` configs.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    # litmodule
    store(
        fs_builds(
            UnconditionalKWGenerationLitModule,
            config=BaseLitModuleConfig(),
        ),
        name="unc_kw_gen",
        group="litmodule",
    )
    store(
        fs_builds(
            KWGenerationLitModule,
            config=KWGenerationLitModuleConfig(),
        ),
        name="kw_gen",
        group="litmodule",
    )
    # litmodule/nnmodule
    store(
        fs_builds(Unet1D, dim=64, dim_mults=(1, 2), channels=1),
        name="unet1d",
        group="litmodule/nnmodule",
    )
    store(
        fs_builds(CustomDiT),
        name="custom_dit",
        group="litmodule/nnmodule",
    )
