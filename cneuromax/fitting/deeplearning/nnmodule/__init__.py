""":class:`torch.nn.Module` Classes."""

from hydra.core.config_store import ConfigStore

from cneuromax.fitting.deeplearning.nnmodule.mlp import MLP, MLPConfig
from cneuromax.utils.hydra import fs_builds

__all__ = ["MLP", "MLPConfig", "store_configs"]


def store_configs(cs: ConfigStore) -> None:
    """Store :mod:`hydra-core` `litmodule/nnmodule` group configs.

    Names: `mlp`.

    Args:
        cs: See\
            :paramref:`cneuromax.config.store_configs.cs`.

    """
    cs.store(
        group="litmodule/nnmodule",
        name="mlp",
        node=fs_builds(MLP, config=MLPConfig()),
    )
