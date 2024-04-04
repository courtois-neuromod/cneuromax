""":class:`CustomDiT` & its helper classes."""

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from x_transformers.x_transformers import AttentionLayers

from cneuromax.projects.kw_pred.dit.models import (
    DiTBlock,
    TimestepEmbedder,
    get_1d_sincos_pos_embed_from_grid,
)


def modulate(
    x: Float[Tensor, " BS NP ES"],
    shift: Float[Tensor, " BS ES"],
    scale: Float[Tensor, " BS ES"],
) -> Float[Tensor, " BS NP ES"]:
    """Modulates (shifts and scales) the input tensor.

    BS: Batch size
    NP: Number of patches
    ES: Embedding size (a.k.a. hidden size)
    """
    pattern = "BS ES -> BS 1 ES"
    shift, scale = rearrange(shift, pattern), rearrange(scale, pattern)
    return x * (1 + scale) + shift


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

        BS: Batch size
        IC: Number of input `.klk` channels
        SL: `.klk` sequence length
        NP: Number of patches
        ES: Embedding size
        """
        x: Float[Tensor, " BS ES NP"] = self.proj(x)
        return rearrange(x, "BS ES NP -> BS NP ES")


class BeatsEmbedder(nn.Module):
    """Custom :attr:`~DiT.y_embedder`.

    Meant to replace :class:`.dit.models.LabelEmbedder`.
    """

    def __init__(
        self: "BeatsEmbedder",
        og_embd_size: int,
        embd_size: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(og_embd_size, embd_size)

        def forward(
            self: "BeatsEmbedder",
            x: Float[Tensor, " BS OES"],
        ) -> Float[Tensor, " BS ES"]:
            """Flattened BEATS -> Embeddings.

            BS: Batch size
            SL: STFT Sequence length (time steps)
            NB: Number of STFT frequency bins
            ES: Embedding size
            NP: Number of patches
            """
            x: Float[Tensor, " BS ES"] = self.proj(x)
            return x


class STFTEmbedder(nn.Module):
    """Custom :attr:`~DiT.y_embedder`.

    Meant to replace :class:`.dit.models.LabelEmbedder` given that our
    conditioning data is the STFT of the audio signal rather than the
    class labels.

    TODO: Debug
    """

    def __init__(  # noqa: PLR0913
        self: "STFTEmbedder",
        seq_len: int,
        num_freq_bins: int,
        embd_size: int,
        num_patches: int,
        encoder: AttentionLayers,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embd_size = embd_size
        self.num_patches = num_patches
        patch_size = seq_len // num_patches
        # STFT -> Patch-wise embedded STFT
        self.patch_embed = PatchEmbed1D(
            input_size=seq_len,
            in_channels=num_freq_bins,
            embd_size=embd_size,
            patch_size=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embd_size),
            requires_grad=False,
        )
        self.encoder = encoder
        self.proj = nn.Linear(num_patches * embd_size, embd_size)

    def forward(
        self: "STFTEmbedder",
        x: Float[Tensor, " BS SL NB"],  # BS x 311 x 513
    ) -> Float[Tensor, " BS ES"]:
        """STFT -> Embeddings.

        BS: Batch size
        SL: STFT Sequence length (time steps)
        NB: Number of STFT frequency bins
        ES: Embedding size
        NP: Number of patches
        """
        x: Float[Tensor, " BS NP ES"] = self.patch_embed(x) + self.pos_embed
        x: Float[Tensor, " BS NP ES"] = self.encoder(x)
        x: Float[Tensor, " BS NPxES"] = rearrange(x, "BS NP ES -> BS (NP ES)")
        x: Float[Tensor, " BS ES"] = self.proj(x)
        return x


class FinalLayer1D(nn.Module):
    """Custom :attr:`~DiT.final_layer`.

    Meant to replace :class:`.dit.models.FinalLayer` given that we use
    1D data instead of 2D data.
    """

    def __init__(
        self: "FinalLayer1D",
        hidden_size: int,
        patch_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * out_channels,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(
        self: "FinalLayer1D",
        x: Float[Tensor, " BS NP ES"],
        c: Float[Tensor, " BS ES"],
    ) -> Float[Tensor, " BS PSxOC"]:
        """Transformer output -> Patch-wise output.

        BS: Batch size
        NP: Number of patches
        ES: Embedding size (a.k.a. hidden size)
        PS: Patch size
        OC: Out channels
        """
        shift, scale = rearrange(
            self.adaLN_modulation(c),
            "BS (split ES) -> split BS ES",
            split=2,
        )
        x: Float[Tensor, " BS NP ES"] = modulate(
            self.norm_final(x),
            shift,
            scale,
        )
        x: Float[Tensor, " BS NP PSxOC"] = self.linear(x)
        return x


class CustomDiT(nn.Module):
    """Custom DiT model."""

    def __init__(  # noqa: PLR0913
        self: "CustomDiT",
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        # class_dropout_prob: float = 0.1,  # noqa: ERA001
        # num_classes: int = 1000,  # noqa: ERA001
        *,
        learn_sigma: bool = True,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        ### NEW ###
        self.input_size = input_size
        ###########
        """
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        """
        ### NEW ###
        self.x_embedder = PatchEmbed1D(
            input_size=input_size,
            in_channels=in_channels,
            embd_size=hidden_size,
            patch_size=patch_size,
        )
        ###########
        # INFO: OpenDIT allows for changing the dtype of `t_embedder`.
        self.t_embedder = TimestepEmbedder(hidden_size)  # type: ignore[no-untyped-call]
        """
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob
        )
        """
        ### NEW ###
        """
        self.y_embedder = STFTEmbedder(  # type: ignore[assignment]
            seq_len=311,
            num_freq_bins=513,
            embd_size=hidden_size,
            num_patches=self.x_embedder.num_patches,
            encoder=AttentionLayers(
                dim=hidden_size,
                depth=depth,
                heads=num_heads,
            ),
        )
        """
        self.y_embedder = BeatsEmbedder(
            og_embd_size=768,
            embd_size=hidden_size,
        )
        ###########
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)  # type: ignore[no-untyped-call]
                for _ in range(depth)
            ],
        )
        """
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels
        )
        """
        ### NEW ###
        self.final_layer = FinalLayer1D(  # type: ignore[assignment]
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=self.out_channels,
        )
        ###########
        self.initialize_weights()

    def initialize_weights(self: "CustomDiT") -> None:  # noqa: D102
        # Initialize transformer layers:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        """
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        """  # noqa: W505
        ### NEW ###
        pos_embed = get_1d_sincos_pos_embed_from_grid(  # type: ignore[no-untyped-call]
            embed_dim=self.pos_embed.shape[-1],
            pos=np.arange(self.x_embedder.num_patches),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )
        """
        pos_embed = get_1d_sincos_pos_embed_from_grid(  # type: ignore[no-untyped-call]
            embed_dim=self.y_embedder.pos_embed.shape[-1],
            pos=np.arange(self.y_embedder.num_patches),
        )
        self.y_embedder.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )
        """
        ###########

        """
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        """  # noqa: W505

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(
        self: "CustomDiT",
        x: Float[Tensor, " BS NP PSxOC"],
    ) -> Float[Tensor, " BS OC SL"]:
        """Converts patch-wise embeddings to 1D data.

        BS: Batch size
        NP: Number of patches
        PS: Patch size
        SL: `.klk` sequence length
        OC: Output channels
        """
        return rearrange(
            x,
            "BS NP (PS OC) -> BS OC (NP PS)",
            OC=self.out_channels,
        )

    def forward(
        self: "CustomDiT",
        x: Float[Tensor, " BS SL IC"],
        t: Int[Tensor, " BS ES"],
        y: Float[Tensor, " BS ES"],
    ) -> Float[Tensor, " BS SL OC"]:
        """.

        BS: Batch size
        SL: `.klk` sequence length
        IC: Number of input `.klk` channels (number of chair corners)
        ES: Embedding size (a.k.a. hidden size)
        OC: Output channels
        NP: Number of patches
        """
        x: Float[Tensor, " BS NP ES"] = self.x_embedder(x) + self.pos_embed
        t: Float[Tensor, " BS ES"] = self.t_embedder(t)
        y: Float[Tensor, " BS ES"] = self.y_embedder(y)
        c: Float[Tensor, " BS ES"] = t + y
        for block in self.blocks:
            x: Float[Tensor, " BS NP ES"] = block(x, c)  # type: ignore[no-redef]
        x: Float[Tensor, " BS PSxOC"] = self.final_layer(x, c)
        x: Float[Tensor, " BS OC SL"] = self.unpatchify(x)
        return x
