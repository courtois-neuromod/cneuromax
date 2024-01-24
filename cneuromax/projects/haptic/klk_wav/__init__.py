"""Data classes for the ``.wav`` files extracted from ``.klk`` files."""
from hydra_zen import ZenStore

from cneuromax.utils.hydra_zen import fs_builds

from .datamodule import KLKWavDataModule, KLKWavDataModuleConfig
from .dataset import KLKWavDataset

__all__ = ["KLKWavDataset", "KLKWavDataModuleConfig", "KLKWavDataModule"]


def store_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``klk_wav`` configs.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        fs_builds(
            KLKWavDataModule,
            config=KLKWavDataModuleConfig(),
        ),
        name="klk_wav",
        group="datamodule",
    )
