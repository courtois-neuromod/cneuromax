""":class:`.BaseLitModule` :mod:`hydra-core` config store."""
from hydra_zen import ZenStore
from torch.optim import SGD, Adam, AdamW
from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

from cneuromax.fitting.deeplearning.litmodule.nnmodule import (
    MLP,
    MLPConfig,
)
from cneuromax.utils.hydra_zen import (
    fs_builds,
    pfs_builds,
)


def store_mlp_config(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/nnmodule`` group config.

    Config name: ``mlp``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        fs_builds(MLP, config=MLPConfig()),
        name="mlp",
        group="litmodule/nnmodule",
    )


def store_basic_optimizer_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/optimizer`` group configs.

    Config names: ``adam``, ``adamw``, ``sgd``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(pfs_builds(Adam), name="adam", group="litmodule/optimizer")
    store(pfs_builds(AdamW), name="adamw", group="litmodule/optimizer")
    store(pfs_builds(SGD), name="sgd", group="litmodule/optimizer")


def store_basic_scheduler_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/scheduler`` group configs.

    Config names: ``constant``, ``linear_warmup``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        pfs_builds(get_constant_schedule),
        name="constant",
        group="litmodule/scheduler",
    )
    store(
        pfs_builds(get_constant_schedule_with_warmup),
        name="linear_warmup",
        group="litmodule/scheduler",
    )
