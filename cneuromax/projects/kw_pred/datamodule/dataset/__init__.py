r""":mod:`.kw_pred` :class:`torch.utils.data.Dataset``s."""

from hydra_zen import ZenStore

from cneuromax.utils.hydra_zen import fs_builds

from .base import KWPredDataset, KWPredDatasetConfig

__all__ = ["KWPredDataset"]


def store_configs(store: ZenStore) -> None:
    """Stores :mod:`.kw_pred` ``datamodule/dataset`` group configs.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        fs_builds(KWPredDataset, config=KWPredDatasetConfig()),
        name="kw_pred",
        group="datamodule/dataset",
    )
