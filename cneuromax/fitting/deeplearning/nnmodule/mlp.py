""":class:`MLP` + its config dataclass & config storing function."""
from dataclasses import dataclass
from typing import Annotated as An

from einops import rearrange
from hydra_zen import ZenStore
from jaxtyping import Float
from omegaconf import MISSING
from torch import Tensor, nn

from cneuromax.utils.annotations import ge, lt
from cneuromax.utils.zen import fs_builds


def store_mlp_config(store: ZenStore) -> None:
    """Stores name ``mlp`` in :mod:`hydra-core` config store.

    Config group: ``litmodule/nnmodule``.

    Config name: ``mlp``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        fs_builds(MLP, config=MLPConfig()),
        name="mlp",
        group="litmodule/nnmodule",
    )


@dataclass
class MLPConfig:
    """Holds :class:`MLP` config values.

    Args:
        dims: List of dimensions for each layer.
        p_dropout: Dropout probability.
    """

    dims: list[int] = MISSING
    p_dropout: An[float, ge(0), lt(1)] = 0.0


class MLP(nn.Module):
    """Multi-layer perceptron (MLP).

    Allows for a variable number of layers, activation functions, and
    dropout probability.

    Args:
        config: See :class:`MLPConfig`.
        activation_fn: The singular activation function to use in\
            between each layer.

    Attributes:
        model (:class:`torch.nn.Sequential`): The internal\
            :class:`~torch.nn.Module` that holds the MLP's layers.
    """

    def __init__(
        self: "MLP",
        config: MLPConfig,
        activation_fn: nn.Module,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential()
        for i in range(len(config.dims) - 1):
            self.model.add_module(
                name=f"fc_{i}",
                module=nn.Linear(config.dims[i], config.dims[i + 1]),
            )
            if i < len(config.dims) - 2:
                self.model.add_module(name=f"act_{i}", module=activation_fn)
                if config.p_dropout:  # > 0.0:
                    self.model.add_module(
                        name=f"drop_{i}",
                        module=nn.Dropout(config.p_dropout),
                    )

    def forward(
        self: "MLP",
        x: Float[Tensor, " batch_size *d_input"],
    ) -> Float[Tensor, " batch_size output_size"]:
        """Flattens input's dimensions and passes it through the model.

        Note:
            This MLP is only capable of returning 1D\
                :class:`torch.Tensor` batches.

        Args:
            x: The input data batch.

        Returns:
            The output batch.
        """
        out: Float[Tensor, " batch_size flattened_d_input"] = rearrange(
            x,
            "batch_size ... -> batch_size (...)",
        )
        out: Float[Tensor, " batch_size output_size"] = self.model(out)
        return out
