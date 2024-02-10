r""":mod:`.kw_pred` :class:`lightning.pytorch.LightningModule``s."""

from hydra_zen import ZenStore

from cneuromax.utils.hydra_zen import pfs_builds


def store_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` :mod:`.kw_pred` configs.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        pfs_builds(KWPredDataset, config=KWPredDatasetConfig()),
        name="kw_pred",
        group="datamodule/dataset",
    )
