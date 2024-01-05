r""":class:`torch.nn.Module`\s."""
from cneuromax.fitting.deeplearning.nnmodule.mlp import (
    MLP,
    MLPConfig,
    store_mlp_config,
)

__all__ = ["MLPConfig", "MLP", "store_mlp_config"]
