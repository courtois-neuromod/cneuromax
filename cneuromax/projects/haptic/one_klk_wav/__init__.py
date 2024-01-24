"""Mini-project using 1 file from :class:`.KLKWavDataset`."""
from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.datamodule import BaseDataModuleConfig
from cneuromax.utils.hydra_zen import fs_builds

from .datamodule import OneKLKWavDataModule
from .dataset import OneKLKWavDataset
from .litmodule import OneKLKWavLitModule

__all__ = ["OneKLKWavDataset", "OneKLKWavDataModule", "OneKLKWavLitModule"]


def store_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``one_klk_wav`` configs.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        fs_builds(OneKLKWavDataModule, config=BaseDataModuleConfig()),
        name="one_klk_wav",
        group="datamodule",
    )
    store(
        fs_builds(OneKLKWavLitModule),
        name="one_klk_wav",
        group="litmodule",
    )
