""":class:`CustomDiT` & its helper classes."""

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from x_transformers.x_transformers import AttentionLayers

from cneuromax.projects.kw_pred.dit.models import (
    DiTBlock,
    FinalLayer,
    TimestepEmbedder,
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

        BS: Batch size
        IC: Number of input `.klk` channels
        SL: `.klk` sequence length
        NP: Number of patches
        ES: Embedding size
        """
        x: Float[Tensor, " BS ES NP"] = self.proj(x)
        return rearrange(x, "BS ES NP -> BS NP ES")


class STFTEmbedder(nn.Module):
    """Custom :attr:`~DiT.y_embedder`.

    Meant to replace :class:`.dit.models.LabelEmbedder` given that our
    conditioning data is the STFT of the audio signal rather than the
    class labels.
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
        *,
        placeholder: bool,  # noqa: ARG002
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
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
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
        self.t_embedder = TimestepEmbedder(hidden_size)  # type: ignore[no-untyped-call]
        """
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob
        )
        """
        ### NEW ###
        self.y_embedder = STFTEmbedder(
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
        self.final_layer = FinalLayer(  # type: ignore[no-untyped-call]
            hidden_size,
            patch_size,
            self.out_channels,
        )
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
        pos_embed = get_1d_sincos_pos_embed_from_grid(  # type: ignore[no-untyped-call]
            embed_dim=self.y_embedder.pos_embed.shape[-1],
            pos=np.arange(self.y_embedder.num_patches),
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

    """ TODO: 2D -> 1D
    def unpatchify(self, x):
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    """

    def forward(self: "CustomDiT", x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass of DiT.
        # x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        x: (N, C, SL) tensor of spatial inputs (haptic)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """  # noqa: D205, D212, D415, W505, E501
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        """
        x = self.unpatchify(x)  # type: ignore[no-untyped-call] # (N, out_channels, H, W)
        """  # noqa: W505, E501
        x = self.unpatchify(x)  # type: ignore[no-untyped-call] # (N, out_channels, SL)
        return x  # type: ignore[no-any-return]  # noqa: RET504
