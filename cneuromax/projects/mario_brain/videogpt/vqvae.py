# Adapted from https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py
""":class:`VQVAE`."""

import argparse
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as f
from torch import nn

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModule,
    BaseLitModuleConfig,
)

from .attention import MultiHeadAttention


@dataclass
class VQVAEConfig(BaseLitModuleConfig):
    """Config for VQVAE."""

    n_hiddens: int = 240
    n_res_layers: int = 4
    downsample: tuple[int] = (4, 16, 16)
    embedding_dim: int = 256
    n_codes: int = 1024
    sequence_length: int = 16
    #    lr_scheduler: str = "CosineAnnealingLR"
    #    lr_scheduler_args: dict | None = None
    #    lr: float = 3e-4
    betas: tuple[float] = (0.9, 0.999)
    resolution: int = 64
    desc: str | None = None
    no_attn: bool = False


class VQVAE(BaseLitModule):
    """VQVAE."""

    def __init__(
        self: "VQVAE",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        # if config.lr_scheduler_args is None:
        #     lr_scheduler_args = {"T_max": 50000}
        super().__init__(*args, **kwargs)
        self.config: VQVAEConfig
        self.sequence_length = self.config.sequence_length
        self.downsample = self.config.downsample
        self.embedding_dim = self.config.embedding_dim
        self.n_codes = self.config.n_codes
        self.resolution = self.config.resolution
        self.desc = self.config.desc
        # self.lr = config.lr
        self.betas = self.config.betas
        # self.lr_scheduler = config.lr_scheduler
        # self.lr_scheduler_args = lr_scheduler_args
        self.val_recon_loss_of_epoch = []

        self.encoder = Encoder(
            self.config.n_hiddens,
            self.config.n_res_layers,
            self.config.downsample,
            not self.config.no_attn,
        )
        self.decoder = Decoder(
            self.config.n_hiddens,
            self.config.n_res_layers,
            self.config.downsample,
            not self.config.no_attn,
        )

        self.pre_vq_conv = SamePadConv3d(
            self.config.n_hiddens, self.config.embedding_dim, kernel_size=1
        )
        self.post_vq_conv = SamePadConv3d(
            self.config.embedding_dim, self.config.n_hiddens, kernel_size=1
        )

        self.codebook = Codebook(
            self.config.n_codes, self.config.embedding_dim
        )
        self.save_hyperparameters()

    @property
    def latent_shape(self: "VQVAE") -> tuple[int]:
        """Dowsampled shape of the latent space (time, with, height)."""
        input_shape = (self.sequence_length, self.resolution, self.resolution)
        return tuple(
            [
                s // d
                for s, d in zip(input_shape, self.downsample, strict=True)
            ],
        )

    def encode(
        self: "VQVAE",
        x: torch.tensor,
        include_embeddings: bool = False,  # noqa: FBT001, FBT002
    ) -> dict[str, torch.Tensor]:
        """Return the codes of the encoded input.

        If `include_embeddings=True`, also return the embeddings.
        """
        h = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(h)
        if include_embeddings:
            return vq_output["encodings"], vq_output["embeddings"]
        return vq_output["encodings"]

    def decode(self, encodings):
        h = f.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(h.movedim(-1, 1))
        return self.decoder(h)

    def forward(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))
        recon_loss = f.mse_loss(x_recon, x) / 0.06

        return recon_loss, x_recon, vq_output

    def step(self, batch, stage):
        if stage == "train":
            return self._training_step(batch)
        return self._validation_step(batch)

    def _training_step(self, batch):
        x = batch["frames"] if type(batch) == dict else batch
        recon_loss, _, vq_output = self.forward(x)
        commitment_loss = vq_output["commitment_loss"]
        loss = recon_loss + commitment_loss
        self.log("tng/recon_loss", recon_loss)
        self.log("tng/perplexity", vq_output["perplexity"])
        self.log("tng/commitment_loss", vq_output["commitment_loss"])
        # self.log("tng/loss", loss)
        return loss

    def _validation_step(self, batch):
        x = batch["frames"] if type(batch) == dict else batch
        recon_loss, _, vq_output = self.forward(x)
        self.log("val/recon_loss", recon_loss, prog_bar=True, sync_dist=True)
        self.log(
            "val/perplexity",
            vq_output["perplexity"],
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/commitment_loss",
            vq_output["commitment_loss"],
            prog_bar=True,
            sync_dist=True,
        )
        self.val_recon_loss_of_epoch.append(recon_loss.item())
        return recon_loss + vq_output["commitment_loss"]

    def on_validation_epoch_end(self):
        mean_val_recon_loss = np.mean(self.val_recon_loss_of_epoch)
        if self.best_val_recon_loss is None:
            self.best_val_recon_loss = mean_val_recon_loss
        else:
            self.best_val_recon_loss = min(
                self.best_val_recon_loss, mean_val_recon_loss
            )
        self.val_recon_loss_of_epoch.clear()

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
    #         self.parameters(), lr=self.lr, betas=self.betas
    #     )
    #     scheduler = getattr(lr_scheduler, self.lr_scheduler)(
    #         optimizer, **self.lr_scheduler_args
    #     )
    #     return [optimizer], [
    #         dict(scheduler=scheduler, interval="step", frequency=1)
    #     ]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False
        )
        parser.add_argument("--embedding_dim", type=int, default=256)
        parser.add_argument("--n_codes", type=int, default=2048)
        parser.add_argument("--n_hiddens", type=int, default=240)
        parser.add_argument("--n_res_layers", type=int, default=4)
        parser.add_argument(
            "--downsample", nargs="+", type=int, default=(4, 4, 4)
        )
        parser.add_argument("--no_attn", action="store_true")
        return parser


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(
            shape=(0,) * 3,
            dim_q=n_hiddens,
            dim_kv=n_hiddens,
            n_head=n_head,
            n_layer=1,
            causal=False,
            attn_type="axial",
        )
        self.attn_w = MultiHeadAttention(
            attn_kwargs=dict(axial_dim=-2), **kwargs
        )
        self.attn_h = MultiHeadAttention(
            attn_kwargs=dict(axial_dim=-3), **kwargs
        )
        self.attn_t = MultiHeadAttention(
            attn_kwargs=dict(axial_dim=-4), **kwargs
        )

    def forward(self, x):
        x = x.movedim(1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = x.movedim(-1, 1)
        return x


class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2),
        )

    def forward(self, x):
        return x + self.block(x)


class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim):
        super().__init__()
        self.register_buffer("embeddings", torch.randn(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = z.movedim(1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = z.movedim(1, -1).flatten(end_dim=-2)
        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = f.one_hot(encoding_indices, self.n_codes).type_as(
            flat_inputs
        )
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = f.embedding(encoding_indices, self.embeddings)
        embeddings = embeddings.movedim(-1, 1)

        commitment_loss = 0.25 * f.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            # Re-init emebddings that are never used
            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z  # For the gradient skip

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        return dict(
            embeddings=embeddings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    def dictionary_lookup(self, encodings):
        embeddings = f.embedding(encodings, self.embeddings)
        return embeddings


class Encoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, downsample, use_attn=True):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        in_channels = 3
        for i in range(max_ds):
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(
                in_channels, n_hiddens, kernel_size=4, stride=stride
            )
            self.convs.append(conv)
            n_times_downsample -= 1
            in_channels = n_hiddens
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)
        self.use_attn = use_attn
        if use_attn:
            self.res_stack = nn.Sequential(
                *[
                    AttentionResidualBlock(n_hiddens)
                    for _ in range(n_res_layers)
                ],
                nn.BatchNorm3d(n_hiddens),
                nn.ReLU(),
            )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = f.relu(conv(h))
        h = self.conv_last(h)
        if self.use_attn:
            h = self.res_stack(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        if use_attn:
            self.res_stack = nn.Sequential(
                *[
                    AttentionResidualBlock(n_hiddens)
                    for _ in range(n_res_layers)
                ],
                nn.BatchNorm3d(n_hiddens),
                nn.ReLU(),
            )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(
                n_hiddens, out_channels, kernel_size=4, stride=us
            )
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.res_stack(x) if self.use_attn else x
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = f.relu(h)
        return h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=True
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since f.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(f.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=True
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            padding=tuple([k - 1 for k in kernel_size]),
        )

    def forward(self, x):
        return self.convt(f.pad(x, self.pad_input))
