""":class:`CustomDiT` & its helper classes."""

import logging
from typing import Annotated as An

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
from cneuromax.utils.beartype import one_of


def modulate(
    x: Float[Tensor, " BS NP ES"],
    shift: Float[Tensor, " BS ES"],
    scale: Float[Tensor, " BS ES"],
) -> Float[Tensor, " BS NP ES"]:
    """Modulates (shifts and scales) the input tensor.

    BS: Batch size
    NP: Number of patches
    ES: Embedding size (a.k.a. hidden size)

    Args:
        x: The input tensor.
        shift: How much to shift the input tensor.
        scale: How much to scale the input tensor.

    Returns:
        The modulated tensor.
    """
    pattern = "BS ES -> BS 1 ES"
    shift, scale = rearrange(shift, pattern), rearrange(scale, pattern)
    return x * (1 + scale) + shift


class PatchEmbed1D(nn.Module):
    """Converts 1D signal(s) into patch-wise embeddings.

    See https://arxiv.org/abs/2010.11929 for more details on patch-wise
    embeddings.

    Meant to replace :class:`~timm.models.vision_transformer.PatchEmbed`
    when using 1D signal(s) instead of 2D signal(s).

    Padding is added to the input signal(s) to ensure that no
    information is lost when converting.

    Args:
        seq_len: The length of the input 1D signal(s).
        num_signals: The number of 1D input signals (e.g., 1 to 4 for\
            ``.klk`` files, 513 for STFT data, ...).
        embd_size: The number of values for each patch-wise\
            embedding.
        patch_size: The length of each patch.
    """

    def __init__(
        self: "PatchEmbed1D",
        seq_len: int,
        num_signals: int,
        embd_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=num_signals,
            out_channels=embd_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=(patch_size - seq_len % patch_size) % patch_size,
        )
        self.num_patches = seq_len // patch_size + int(
            seq_len % patch_size > 0,
        )

    def forward(
        self: "PatchEmbed1D",
        x: Float[Tensor, " BS NS SL"],
    ) -> Float[Tensor, " BS NP ES"]:
        """1D Data -> Patch-wise embeddings.

        BS: Batch size
        NS: Number of 1D input signals
        SL: The length of the input 1D signal(s)
        NP: Number of patches
        ES: Embedding size
        """
        x: Float[Tensor, " BS ES NP"] = self.proj(x)
        return rearrange(x, "BS ES NP -> BS NP ES")


class TransformerEncode1D(nn.Module):
    """Encodes 1D conditioning signal(s) w/ a transformer encoder.

    Meant to replace :class:`.dit.models.LabelEmbedder` when the
    conditioning data is made up of 1D signals.

    Args:
        seq_len: See :paramref:`PatchEmbed1D.seq_len`.
        num_signals: See :paramref:`PatchEmbed1D.num_signals`.
        embd_size: See :paramref:`PatchEmbed1D.embd_size`.
        x_embedder_num_patches: Self-explanatory.
        encoder: The transformer encoder.
    """

    def __init__(  # noqa: PLR0913
        self: "TransformerEncode1D",
        seq_len: int,
        num_signals: int,
        embd_size: int,
        x_embedder_num_patches: int,
        encoder: AttentionLayers,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embd_size = embd_size
        patch_size = seq_len // x_embedder_num_patches or 1
        self.patch_embed = PatchEmbed1D(
            seq_len=seq_len,
            num_signals=num_signals,
            embd_size=embd_size,
            patch_size=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embd_size),
            requires_grad=False,
        )
        self.encoder = encoder
        self.proj = nn.Linear(
            self.patch_embed.num_patches * embd_size,
            embd_size,
        )

    def forward(
        self: "TransformerEncode1D",
        x: Float[Tensor, " BS SL NS"],
    ) -> Float[Tensor, " BS ES"]:
        """1D signal(s) -> Encoded conditioning information.

        BS: Batch size
        SL: The length of the input 1D signal(s)
        NS: Number of 1D input signals
        ES: Embedding size
        NP: Number of patches
        """
        x: Float[Tensor, " BS NS SL"] = rearrange(x, "BS SL NS -> BS NS SL")
        x: Float[Tensor, " BS NP ES"] = self.patch_embed(x) + self.pos_embed
        x: Float[Tensor, " BS NP ES"] = self.encoder(x)
        x: Float[Tensor, " BS NPxES"] = rearrange(x, "BS NP ES -> BS (NP ES)")
        x: Float[Tensor, " BS ES"] = self.proj(x)
        return x


class FinalLayer1D(nn.Module):
    """Custom :attr:`~DiT.final_layer`.

    Meant to replace :class:`.dit.models.FinalLayer` when using 1D
    signal(s) instead of 2D signal(s).

    Args:
        embd_size: See :paramref:`PatchEmbed1D.embd_size`.
        patch_size: See :paramref:`PatchEmbed1D.patch_size`.
        out_channels: See :attr:`CustomDiT.out_channels`.
    """

    def __init__(
        self: "FinalLayer1D",
        embd_size: int,
        patch_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            embd_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            embd_size,
            patch_size * out_channels,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embd_size, 2 * embd_size, bias=True),
        )

    def forward(
        self: "FinalLayer1D",
        x: Float[Tensor, " BS NP ES"],
        c: Float[Tensor, " BS ES"],
    ) -> Float[Tensor, " BS NP PSxOC"]:
        """Transformer output -> Patch-wise output.

        BS: Batch size
        NP: Number of patches
        ES: Embedding size
        PS: Patch size
        OC: Output channels
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
    """Custom DiT model.

    Args:
        input_size: The length of the input 1D signal.
        patch_size: The length of each patch.
        in_channels: The number of input channels (e.g., 4 for `.klk`\
            files, 513 for STFT data, ...).
        depth: The number of transformer blocks.
        num_heads: The number of attention heads.
        mlp_ratio: By how much to scale the hidden size in the\
            MLP sub-blocks.
        conditioning: How to process the conditioning data. Either a
            linear layer or a transformer encoder.
        audio_stft_rel_dir: See\
            :paramref:`.KWPredDatasetConfig.audio_stft_rel_dir`.
        audio_embeddings_rel_dir: See\
            :paramref:`.KWPredDatasetConfig.audio_embeddings_rel_dir`.
        learn_sigma: Whether to learn the standard deviation of the\
            output distribution.
    """

    def __init__(  # noqa: PLR0913
        self: "CustomDiT",
        ### NEW ###
        klk_seq_len: int = 4000,  # replaces `input_size: int = 32,`
        num_klk_corners: int = 1,  # replaces `in_channels: int = 4,`
        conditioning: An[str, one_of("linear", "transformer")] = "linear",
        audio_stft_rel_dir: str | None = None,
        audio_embeddings_rel_dir: str | None = None,
        ###########
        patch_size: int = 2,
        # hidden_size: int = 1152, now `embd_size` & computed w/ `depth`
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
        self.num_klk_corners = num_klk_corners
        self.out_channels = (
            num_klk_corners * 2 if learn_sigma else num_klk_corners
        )
        self.patch_size = patch_size
        self.num_heads = num_heads
        ### NEW ###
        self.klk_seq_len = klk_seq_len
        self.embd_size = 64 * depth  # As in MM-DiT paper (old `hidden_size`)
        if bool(audio_embeddings_rel_dir) == bool(audio_stft_rel_dir):
            error_msg = (
                "Exactly one of `audio_embeddings_rel_dir` and "
                "`audio_stft_rel_dir` must be provided."
            )
            raise ValueError(error_msg)
        ###########
        """
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        """
        ### NEW ###
        self.x_embedder = PatchEmbed1D(
            seq_len=num_klk_corners,
            num_signals=klk_seq_len,
            embd_size=self.embd_size,
            patch_size=patch_size,
        )
        ###########
        # INFO: OpenDIT allows for changing the dtype of `t_embedder`.
        self.t_embedder = TimestepEmbedder(self.embd_size)  # type: ignore[no-untyped-call]
        """
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob
        )
        """
        ### NEW ###
        # BEATS: 62 (8) 768
        # STFT: 311 513
        conditioning_seq_len = 62 if audio_embeddings_rel_dir else 311
        conditioning_num_signals = 768 if audio_embeddings_rel_dir else 513
        self.y_embedder: nn.Module
        if conditioning == "linear":
            self.y_embedder = nn.Linear(
                conditioning_num_signals,
                self.embd_size,
            )
        else:  # conditioning == "transformer"
            self.y_embedder = TransformerEncode1D(
                seq_len=conditioning_seq_len,
                num_signals=conditioning_num_signals,
                embd_size=self.embd_size,
                x_embedder_num_patches=self.x_embedder.num_patches,
                encoder=AttentionLayers(
                    dim=self.embd_size,
                    depth=depth,
                    heads=num_heads,
                ),
            )
        ###########
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embd_size),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(self.embd_size, num_heads, mlp_ratio=mlp_ratio)  # type: ignore[no-untyped-call]
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
            embd_size=self.embd_size,
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
            embed_dim=self.embd_size,
            pos=np.arange(self.x_embedder.num_patches),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )
        if isinstance(self.y_embedder, TransformerEncode1D):
            pos_embed = get_1d_sincos_pos_embed_from_grid(  # type: ignore[no-untyped-call]
                embed_dim=self.embd_size,
                pos=np.arange(self.y_embedder.patch_embed.num_patches),
            )
            self.y_embedder.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0),
            )
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
        x: Float[Tensor, " BS IC SL"],
        t: Int[Tensor, " BS"],
        y: (
            Float[Tensor, " BS AES"]
            | Float[Tensor, " BS NAE AES"]
            | Float[Tensor, " BS ES"]
        ),
    ) -> Float[Tensor, " BS SL OC"]:
        """.

        BS: Batch size
        IC: Number of input `.klk` channels (number of chair corners)
        SL: `.klk` sequence length
        ES: Embedding size (a.k.a. hidden size)
        OC: Output channels
        NP: Number of patches
        AES: Audio embeddings size
        NAE: Number of audio embeddings (time dimension)
        """
        x: Float[Tensor, " BS NP ES"] = self.x_embedder(x) + self.pos_embed
        t: Float[Tensor, " BS ES"] = self.t_embedder(t)
        y: Float[Tensor, " BS ES"] = self.y_embedder(y)
        c: Float[Tensor, " BS ES"] = t + y
        for block in self.blocks:
            x: Float[Tensor, " BS NP ES"] = block(x, c)  # type: ignore[no-redef]
        x: Float[Tensor, " BS NP PSxOC"] = self.final_layer(x, c)
        x: Float[Tensor, " BS OC SL"] = self.unpatchify(x)
        return x
