"""MLP config & class."""

from dataclasses import dataclass
from typing import Annotated as An

from einops import rearrange
from jaxtyping import Float
from omegaconf import MISSING
from torch import Tensor, nn

from cneuromax.utils.annotations import ge, lt


@dataclass
class MLPConfig:
    """Config for :class:`MLP` instances.

    Attributes:
        dims: List of dimensions for each layer.
        p_dropout: Dropout probability.
    """

    dims: list[int] = MISSING
    p_dropout: An[float, ge(0), lt(1)] = 0.0


class MLP(nn.Module):
    """Multi-layer perceptron (MLP).

    Allows for a variable number of layers, activation functions, and
    dropout probability.

    Attributes:
        config (:class:`MLPConfig`): This instance's configuration,\
            see :class:`MLPConfig`.
        model (:class:`~torch.nn.Sequential`): The MLP model.
    """

    def __init__(
        self: "MLP",
        config: MLPConfig,
        activation_fn: nn.Module,
    ) -> None:
        """Calls parent constructor & initializes model.

        Args:
            config: This instance's configuration, see\
                :class:`MLPConfig`.
            activation_fn: Activation function to use between layers.
        """
        super().__init__()
        self.config = config
        self.model = nn.Sequential()

        for i in range(len(config.dims) - 1):
            self.model.add_module(
                f"fc_{i}",
                nn.Linear(config.dims[i], config.dims[i + 1]),
            )
            if i < len(config.dims) - 2:
                self.model.add_module(f"act_{i}", activation_fn)
                if config.p_dropout:  # > 0.0:
                    self.model.add_module(
                        f"drop_{i}",
                        nn.Dropout(config.p_dropout),
                    )

    def forward(
        self: "MLP",
        x: Float[Tensor, " batch_size *d_input"],
    ) -> Float[Tensor, " batch_size output_size"]:
        """Flattens input dimensions and pass through the model.

        Note:
            This MLP isn't currently suitable for cases where the output
            is multidimensional.

        Args:
            x: The data input batch.

        Returns:
            The output batch.
        """
        out: Float[Tensor, " batch_size flattened_d_input"] = rearrange(
            x,
            "batch_size ... -> batch_size (...)",
        )
        out: Float[Tensor, " batch_size output_size"] = self.model(out)
        return out
