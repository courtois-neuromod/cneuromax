"""."""

from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class MLP(nn.Module):
    """Multi-layer perceptron (MLP).

    Allows for a variable number of layers, activation functions, and
    dropout probability.

    Attributes:
        model: The MLP model.
    """

    def __init__(
        self: "MLP",
        dims: list[int],
        activation_fn: type[nn.Module] = nn.ReLU,
        p_dropout: float = 0.0,
    ) -> None:
        """Calls parent constructor and initializes model.

        Args:
            dims: List of dimensions for each layer.
            activation_fn: Activation function.
            p_dropout: Dropout probability.
        """
        super().__init__()

        self.model = nn.Sequential()

        for i in range(len(dims) - 1):
            self.model.add_module(f"fc_{i}", nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.model.add_module(f"act_{i}", activation_fn())
                if p_dropout:  # > 0.0:
                    self.model.add_module(f"drop_{i}", nn.Dropout(p_dropout))

    def forward(
        self: "MLP",
        x: Float[Tensor, " batch_size *d_input"],
    ) -> Float[Tensor, " batch_size output_size"]:
        """Flattens input dimensions and pass through the model.

        Note:
            This MLP isn't (yet?) suitable for cases where the output is
            multidimensional.

        Args:
            x: .

        Returns:
            The output vector batch.
        """
        x = rearrange(x, "batch_size ... -> batch_size (...)")
        x: Tensor = self.model(x)
        return x
