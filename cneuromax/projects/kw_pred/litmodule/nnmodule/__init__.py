r""":mod:`.kw_pred` :class:`torch.nn.Module``s."""

from hydra_zen import ZenStore


def store_configs(store: ZenStore) -> None:
    """Stores :mod:`.kw_pred` ``datamodule/dataset`` group configs.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
