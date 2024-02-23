"""Deep Learning :mod:`hydra-core` config store."""

from hydra_zen import ZenStore
from lightning.pytorch import Trainer

from cneuromax.utils.hydra_zen import (
    pfs_builds,
)


def store_basic_trainer_config(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``trainer`` group configs.

    Config name: ``base``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        pfs_builds(
            Trainer,
            accelerator="${config.device}",
            default_root_dir="${config.output_dir}/lightning/",
            gradient_clip_val=1.0,
        ),
        name="base",
        group="trainer",
    )
