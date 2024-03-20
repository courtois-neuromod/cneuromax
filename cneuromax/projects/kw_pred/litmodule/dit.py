"""."""

from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor, nn

from cneuromax.projects.kw_pred.dit.models import (
    DiT,
    get_1d_sincos_pos_embed_from_grid,
)


class OneDPatchEmbed(nn.Module):
    """Custom :attr:`.DiT.x_embedder`.

    Meant to replace :class:`timm.models.vision_transformer.PatchEmbed`
    given that we input 1D data as opposed to 2D images.
    """

    def __init__(  # noqa: PLR0913
        self: "OneDPatchEmbed",
        input_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.num_patches = input_size // kernel_size

    def forward(
        self: "OneDPatchEmbed",
        x: Float[Tensor, " batch_size in_channels seq_len"],
    ) -> Float[Tensor, " batch_size num_patches out_channels"]:
        """Original input -> Input for the transformer."""
        x: Float[Tensor, " batch_size out_channels num_patches"] = self.proj(x)
        x: Float[Tensor, " batch_size num_patches out_channels"] = x.transpose(
            dim0=-1,
            dim1=-2,
        )
        return x


class STFTEmbedder(nn.Module):
    """Custom :attr:`~DiT.y_embedder`.

    Meant to replace :class:`.dit.models.LabelEmbedder` given that our
    conditioning data is the STFT of the audio signal rather than the
    class labels.
    """

    def __init__(self: "STFTEmbedder") -> None:
        super().__init__()
        self.embedding_table = None

    def forward(
        self: "STFTEmbedder",
        x: Float[Tensor, " batch_size in_channels seq_len"],
    ) -> Float[Tensor, " batch_size num_patches out_channels"]:
        """Forward pass."""
        return x


class CustomDiT(DiT):
    """Custom DiT model."""

    def __init__(
        self: "CustomDiT",
        input_size: int = 32,
        hidden_size: int = 1152,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the model."""
        super().__init__(input_size, hidden_size, *args, **kwargs)  # type: ignore[no-untyped-call]
        self.hidden_size = hidden_size
        self.x_embedder = OneDPatchEmbed(
            input_size=input_size,
            in_channels=self.in_channels,
            out_channels=hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def initialize_weights(self: "CustomDiT") -> None:  # noqa: D102
        super().initialize_weights()  # type: ignore[no-untyped-call]
        pos_embed = get_1d_sincos_pos_embed_from_grid(  # type: ignore[no-untyped-call]
            embed_dim=self.hidden_size,
            pos=self.x_embedder.num_patches,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )
