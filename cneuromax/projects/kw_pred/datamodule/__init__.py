r""":mod:`.kw_pred` :class:`lightning.pytorch.LightningDataModule``s."""

from hydra_zen import ZenStore

from cneuromax.utils.hydra_zen import fs_builds

from .base import KWPredDataModule, KWPredDatamoduleConfig
from .dataset import store_configs as store_dataset_configs


def store_configs(store: ZenStore) -> None:
    """Stores :mod:`.kw_pred` ``datamodule`` group configs.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        fs_builds(KWPredDataModule, config=KWPredDatamoduleConfig()),
        name="kw_pred",
        group="datamodule",
    )
    store_dataset_configs(store=store)
