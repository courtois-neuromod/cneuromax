r"""Common :class:`torch.nn.Module`\s (+:mod:`hydra-core` cfg store)."""

from hydra.core.config_store import ConfigStore

from cneuromax.fitting.deeplearning.nnmodule.mlp import MLP, MLPConfig
from cneuromax.utils.hydra import fs_builds

__all__ = ["MLP", "MLPConfig", "store_nnmodule_configs"]


def store_nnmodule_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/nnmodule`` group configs.

    Config names: ``mlp``.

    Args:
        cs: See :paramref:`cneuromax.config.store_task_configs.cs`.
    """
    cs.store(
        group="litmodule/nnmodule",
        name="mlp",
        node=fs_builds(MLP, config=MLPConfig()),
    )
