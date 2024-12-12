""":class:`.BaseLitModule` `Hydra <https://hydra.cc>`_ config store."""

from hydra_zen import ZenStore
from schedulefree import AdamWScheduleFree
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


def store_basic_nnmodule_config(store: ZenStore) -> None:
    """Stores ``hydra`` ``litmodule/nnmodule`` group config.

    Ref: `Hydra <https://hydra.cc>`_

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
    """Stores ``hydra`` ``litmodule/optimizer`` group configs.

    Ref: `Hydra <https://hydra.cc>`_.

    Config names: ``adam``, ``adamw``, ``sgd``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(pfs_builds(Adam), name="adam", group="litmodule/optimizer")
    store(pfs_builds(AdamW), name="adamw", group="litmodule/optimizer")
    store(
        pfs_builds(AdamWScheduleFree),
        name="sfadamw",
        group="litmodule/optimizer",
    )
    store(pfs_builds(SGD), name="sgd", group="litmodule/optimizer")


def store_basic_scheduler_configs(store: ZenStore) -> None:
    """Stores ``hydra`` ``litmodule/scheduler`` group configs.

    Ref: `Hydra <https://hydra.cc>`_.

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
