""":class:`CustomDiT` & its helper classes."""

from typing import Any

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from cneuromax.projects.kw_pred.dit.models import (
    DiT,
    get_1d_sincos_pos_embed_from_grid,
)


class PatchEmbed1D(nn.Module):
    """Converts a 1D signal into patch-wise embeddings.

    Meant for :attr:`.DiT.x_embedder` originally a
    :class:`~timm.models.vision_transformer.PatchEmbed` instance
    that converts a 2D image into patch-wise embeddings.
    """

    def __init__(
        self: "PatchEmbed1D",
        input_size: int,
        in_channels: int,
        embd_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embd_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.num_patches = input_size // patch_size

    def forward(
        self: "PatchEmbed1D",
        x: Float[Tensor, " BS IC SL"],
    ) -> Float[Tensor, " BS NP ES"]:
        """1D Data -> Patch-wise embeddings.

        BS: batch size
        IC: number of input channels
        SL: sequence length
        NP: number of patches
        ES: embedding size
        """
        x: Float[Tensor, " BS ES NP"] = self.proj(x)
        return rearrange(x, "BS ES NP -> BS NP ES")


class STFTEmbedder(nn.Module):
    """Custom :attr:`~DiT.y_embedder`.

    Meant to replace :class:`.dit.models.LabelEmbedder` given that our
    conditioning data is the STFT of the audio signal rather than the
    class labels.

    Adds
    """

    def __init__(
        self: "STFTEmbedder",
        stft_size: int,
        seq_len: int,
        embd_size: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=stft_size, out_features=embd_size)
        self.init_pos_embed()
        # TODO: x + pos_embd -> wx + pos_embd
        # TODO: init weights

    def init_pos_embed(self: "STFTEmbedder") -> None:
        """Initialize positional embeddings."""
        pos_embed = get_1d_sincos_pos_embed_from_grid(  # type: ignore[no-untyped-call]
            embed_dim=embd_size,
            pos=np.arange(seq_len),
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, seq_len, embd_size),
            requires_grad=False,
        )

        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )

    def forward(
        self: "STFTEmbedder",
        x: Float[Tensor, " batch_size in_channels seq_len"],
        *,
        placeholder: bool,  # noqa: ARG002
    ) -> Float[Tensor, " batch_size num_patches out_channels"]:
        """Forward pass."""
        return self.proj(x) + self.pos_embed


class CustomDiT(DiT):
    """Custom DiT model."""

    def __init__(
        self: "CustomDiT",
        input_size: int = 32,
        patch_size: int = 1,
        in_channels: int = 1,
        hidden_size: int = 1152,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        embd_size = hidden_size
        """Initialize the model."""
        self.x_embedder = PatchEmbed1D(
            input_size=input_size,
            in_channels=in_channels,
            embd_size=embd_size,
            patch_size=patch_size,
        )
        super().__init__(  # type: ignore[no-untyped-call]
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            *args,
            **kwargs,
        )

    def initialize_weights(self: "CustomDiT") -> None:  # noqa: D102
        super().initialize_weights()  # type: ignore[no-untyped-call]
        pos_embed = get_1d_sincos_pos_embed_from_grid(  # type: ignore[no-untyped-call]
            embed_dim=self.hidden_size,
            pos=np.arange(self.x_embedder.num_patches),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )

    def unpatchify(self: "CustomDiT", x: Tensor) -> Tensor:  # noqa: D102
        return x
