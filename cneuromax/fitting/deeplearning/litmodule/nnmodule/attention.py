""":class:`Attention` & :class:`AttentionConfig`.

:mod:`jaxtyping` & :mod:`einops` notation:
- BS: Batch size.
- ES: Embedding size (:paramref:`~AttentionConfig.embd_size`).
- NH: Number of heads (:paramref:`~AttentionConfig.num_heads`).
- HS: Head size (:paramref:`~AttentionConfig.head_size`).
- SL: Sequence length.
- MSL: Max sequence length\
    (:paramref:`~AttentionConfig.max_seq_len`).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as f
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


@dataclass
class AttentionConfig:
    """Holds :class:`Attention` config values.

    Args:
        embd_size: Size of each embedding vector.
        num_heads: Number of attention heads.
        head_size: Size of each attention head vector.
        max_seq_len: Maximum context length.
        bias: Whether to include a bias in the linear layers.
        flash: Whether to use the ``flash attention`` mechanism.
        is_causal: Whether a position is allowed to attend to positions
            subsequent to it.
        p_dropout: Dropout probability.
    """

    embd_size: int
    num_heads: int
    head_size: int
    max_seq_len: int
    bias: bool = False
    flash: bool = True
    is_causal: bool = False
    p_dropout: float = 0.1

    def __post_init__(self: "AttentionConfig") -> None:
        """Verifies the config values."""
        if self.embd_size != self.num_heads * self.head_size:
            error_msg: str = (
                "`embd_size` must be equal to `num_heads * head_size`."
            )
            raise ValueError(error_msg)


class Attention(nn.Module):
    """Scaled dot-product attention.

    Args:
        config: See :class:`AttentionConfig`.
    """

    def __init__(self: "Attention", config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.fc_qkv = nn.Linear(
            in_features=config.embd_size,
            # Combine the weights for the query, key and value
            out_features=3 * config.embd_size,
            bias=config.bias,
        )
        self.fc_proj = nn.Linear(
            in_features=config.embd_size,
            out_features=config.embd_size,
            bias=config.bias,
        )
        self.dropout_1 = nn.Dropout(p=config.p_dropout)
        self.dropout_2 = nn.Dropout(p=config.p_dropout)
        if not config.flash and config.is_causal:
            # Create causal mask
            ones: Float[Tensor, " MSL MSL"] = torch.ones(
                size=(config.max_seq_len, config.max_seq_len),
            )
            trilled_ones: Float[Tensor, " MSL MSL"] = torch.tril(input=ones)
            trilled_ones_reshaped: Float[Tensor, " 1 1 MSL MSL"] = rearrange(
                tensor=trilled_ones,
                pattern="MSL1 MSL2 -> 1 1 MSL1 MSL2",
            )
            self.register_buffer(name="mask", tensor=trilled_ones_reshaped)

    def forward(
        self: "Attention",
        x: Float[Tensor, " BS SL ES"],
    ) -> Float[Tensor, " BS SL ES"]:
        """Forward pass of the attention layer.

        Args:
            x: Input batch.

        Returns:
            Output batch.
        """
        seq_len: int = x.shape[1]
        qkv: Float[Tensor, " BS SL ESx3"] = self.fc_qkv(x)
        qkv_rearranged: Float[Tensor, " 3 BS NH SL HS"] = rearrange(
            tensor=qkv,
            pattern="BS SL (NH HS three) -> three BS NH SL HS",
            head_size=self.config.head_size,
            num_heads=self.config.num_heads,
        )
        q, k, v = qkv_rearranged
        if self.config.flash:
            f.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.config.p_dropout if self.training else 0,
                is_causal=self.config.is_causal,
            )
        else:
            k_t: Float[Tensor, " BS NH HS SL"] = rearrange(
                tensor=k,
                pattern="BS NH SL HS -> BS NH HS SL",
            )
            # BS NH SL HS x BS NH HS SL -> BS NH SL SL
            q_k: Float[Tensor, " BS NH SL SL"] = q @ k_t
            q_k_scaled: Float[Tensor, " BS NH SL SL"] = q_k / (
                self.config.embd_size**0.5
            )
            if self.config.is_causal:
                q_k_scaled.masked_fill(
                    mask=self.mask[:, :, :seq_len, :seq_len] == 0,
                    value=float("-inf"),
                )
            softmax_q_k: Float[Tensor, " BS NH SL SL"] = f.softmax(
                input=q_k_scaled,
                dim=-1,
            )
            softmax_q_k_dropped: Float[Tensor, " BS NH SL SL"] = (
                self.dropout_1(
                    input=softmax_q_k,
                )
            )
            # BS NH SL SL x BS NH SL HS -> BS NH SL HS
            attention_q_k_v: Float[Tensor, " BS NH SL HS"] = (
                softmax_q_k_dropped @ v
            )
        attention_q_k_v_rearranged: Float[Tensor, " BS SL ES"] = rearrange(
            tensor=attention_q_k_v,
            pattern="BS NH SL HS -> BS SL (NH HS)",
        )
        attention_q_k_v_rearranged_dropped: Float[Tensor, " BS SL, ES"] = (
            self.dropout_2(
                input=attention_q_k_v_rearranged,
            )
        )
        return attention_q_k_v_rearranged_dropped
