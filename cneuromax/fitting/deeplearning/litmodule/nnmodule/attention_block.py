""":class:`AttentionBlock`.

:mod:`jaxtyping` & :mod:`einops` notation:
- BS: Batch size.
- ES: Embedding size (:paramref:`~AttentionConfig.embd_size`).
- NH: Number of heads (:paramref:`~AttentionConfig.num_heads`).
- HS: Head size (:paramref:`~AttentionConfig.head_size`).
- SL: Sequence length.
- MSL: Max sequence length\
    (:paramref:`~AttentionConfig.max_seq_len`).
"""

from jaxtyping import Float
from torch import Tensor, nn

from .attention import Attention, AttentionConfig


class AttentionBlock(nn.Module):
    """Attention block.

    Args:
        config: See :class:`AttentionConfig`.
    """

    def __init__(self: "AttentionBlock", config: "AttentionConfig") -> None:
        super().__init__()
        self.config = config
        self.module_1 = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=config.embd_size,
                bias=config.bias,
            ),
            Attention(config=config),
        )
        self.module_2 = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=config.embd_size,
                bias=config.bias,
            ),
            AttentionBlock.MLP(
                embd_size=config.embd_size,
                p_dropout=config.p_dropout,
                bias=config.bias,
            ),
        )

    def forward(
        self: "AttentionBlock",
        x: Float[Tensor, " BS SL ES"],
    ) -> Float[Tensor, " BS SL ES"]:
        """Forward pass of the attention block.

        Args:
            x: Input batch.

        Returns:
            Output batch.
        """
        x: Float[Tensor, " BS SL ES"] = x + self.module_1(x)
        x: Float[Tensor, " BS SL ES"] = x + self.module_2(x)
        return x

    class MLP(nn.Module):
        """Multi-layer perceptron (MLP) for the attention block.

        Args:
            embd_size: See :paramref:`~.AttentionConfig.embd_size`.
            p_dropout: See :paramref:`~.AttentionConfig.p_dropout`.
            bias: See :paramref:`~.AttentionConfig.bias`.

        Attributes:
            module (:class:`torch.nn.Sequential`): MLP module.
        """

        def __init__(
            self: "AttentionBlock.MLP",
            embd_size: int,
            p_dropout: float,
            *,
            bias: bool,
        ) -> None:
            super().__init__()
            self.module = nn.Sequential(
                nn.Linear(
                    in_features=embd_size,
                    out_features=4 * embd_size,
                    bias=bias,
                ),
                nn.GELU(),
                nn.Linear(
                    in_features=4 * embd_size,
                    out_features=embd_size,
                    bias=bias,
                ),
                nn.Dropout(p=p_dropout),
            )

        def forward(  # noqa: D102
            self: "AttentionBlock.MLP",
            x: Float[Tensor, " BS SL ES"],
        ) -> Float[Tensor, " BS SL ES"]:
            x: Tensor = self.module(x)
            return x
