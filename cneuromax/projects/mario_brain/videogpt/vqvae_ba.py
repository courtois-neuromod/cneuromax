# Adapted from https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py
import numpy as np
import argparse
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.distributed as dist

from src.videogpt.attention import MultiHeadAttention


class VQVAE(pl.LightningModule):
    def __init__(
        self,
        sequence_length,
        n_hiddens,
        n_res_layers,
        downsample,
        embedding_dim,
        n_codes,
        n_voxels,
        brain_loss_lambda,
        lr_scheduler="CosineAnnealingLR",
        lr_scheduler_args={"T_max": 50000},
        lr=3e-4,
        weight_decay=0,
        betas=(0.9, 0.999),
        resolution=64,
        brain_dropout=0,
        no_attn=False,
        frozen_encoding=False,
        encoder_rank=None,
        desc=None,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.downsample = downsample
        self.embedding_dim = embedding_dim
        self.n_codes = n_codes
        self.resolution = resolution
        self.desc = desc
        self.lr = lr
        self.weight_decay = weight_decay
        self.brain_loss_lambda = brain_loss_lambda
        self.betas = betas
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args

        self.encoder = Encoder(n_hiddens, n_res_layers, downsample, not no_attn)
        self.decoder = Decoder(n_hiddens, n_res_layers, downsample, not no_attn)

        self.pre_vq_conv = SamePadConv3d(n_hiddens, embedding_dim, kernel_size=1)
        self.post_vq_conv = SamePadConv3d(embedding_dim, n_hiddens, kernel_size=1)

        # self.brain_enc_conv = nn.Conv1d(n_seq_per_bold, 1, n_seq_per_bold)
        flatten_emb_shape = embedding_dim * np.prod(self.latent_shape)
        self.brain_enc_dropout = nn.Dropout(p=brain_dropout)
        if encoder_rank is None:
            self.brain_encoding = nn.Linear(
                in_features=flatten_emb_shape, out_features=n_voxels
            )
        else:
            self.brain_encoding = LowRankEncoder(
                in_features=flatten_emb_shape, out_features=n_voxels, rank=encoder_rank
            )
        self.encoder_rank = encoder_rank
        if frozen_encoding:
            for param in self.brain_encoding.parameters():
                param.requires_grad = False
        self.frozen_encoding = frozen_encoding
        self.n_voxels = n_voxels

        self.codebook = Codebook(n_codes, embedding_dim)
        self.save_hyperparameters()

    @property
    def latent_shape(self):
        input_shape = (self.sequence_length, self.resolution, self.resolution)
        return tuple([s // d for s, d in zip(input_shape, self.downsample)])

    def encode(self, x, include_embeddings=False):
        h = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(h)
        if include_embeddings:
            return vq_output["encodings"], vq_output["embeddings"]
        else:
            return vq_output["encodings"]

    def decode(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(h.movedim(-1, 1))
        return self.decoder(h)

    def forward(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))
        recon_loss = F.mse_loss(x_recon, x) / 0.06

        return recon_loss, x_recon, vq_output

    def training_step(self, batch, batch_idx):
        x = batch["frames"]
        batch_size = x.shape[0]
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        recon_loss, _, vq_output = self.forward(x)
        commitment_loss = vq_output["commitment_loss"]
        embedding = vq_output["embeddings"].reshape(
            batch_size, -1, *vq_output["embeddings"].shape[1:]
        )
        loss_lambda = self.brain_loss_lambda
        if self.brain_loss_lambda == 0:
            embedding = embedding.detach()
            loss_lambda = 1
        brain_encoding = self.brain_encoding(
            self.brain_enc_dropout(embedding.mean(dim=1).flatten(start_dim=1))
        )
        brain_loss = F.mse_loss(brain_encoding, batch["bold"][:, : self.n_voxels])
        loss = (recon_loss + commitment_loss) * (1 - self.brain_loss_lambda)
        loss += loss_lambda * brain_loss
        self.log("tng/recon_loss", recon_loss)
        self.log("tng/brain_loss", brain_loss)
        self.log("tng/perplexity", vq_output["perplexity"])
        self.log("tng/commitment_loss", vq_output["commitment_loss"])
        self.log("tng/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["frames"]
        batch_size = x.shape[0]
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        recon_loss, _, vq_output = self.forward(x)
        embedding = vq_output["embeddings"].reshape(
            batch_size, -1, *vq_output["embeddings"].shape[1:]
        )
        brain_encoding = self.brain_encoding(embedding.mean(dim=1).flatten(start_dim=1))
        brain_loss = F.mse_loss(brain_encoding, batch["bold"][:, : self.n_voxels])
        self.log("val/recon_loss", recon_loss, prog_bar=True, sync_dist=True)
        self.log("val/brain_loss", brain_loss, prog_bar=True, sync_dist=True)
        self.log(
            "val/recon_and_brain_loss",
            self.brain_loss_lambda * brain_loss + recon_loss,
            sync_dist=True,
        )
        self.log(
            "val/perplexity", vq_output["perplexity"], prog_bar=True, sync_dist=True
        )
        self.log(
            "val/commitment_loss",
            vq_output["commitment_loss"],
            prog_bar=True,
            sync_dist=True,
        )
        return {
            "recon_loss": recon_loss,
            "brain_loss": brain_loss,
            "perplexity": vq_output["perplexity"],
            "commitment_loss": vq_output["commitment_loss"],
            "recon_and_brain_loss": self.brain_loss_lambda * brain_loss + recon_loss,
        }

    def configure_optimizers(self):
        param_list = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
            {"params": self.pre_vq_conv.parameters()},
            {"params": self.post_vq_conv.parameters()},
        ]
        if not self.frozen_encoding:
            param_list.append(
                {
                    "params": self.brain_encoding.parameters(),
                    "weight_decay": self.weight_decay,
                },
            )
        optimizer = torch.optim.Adam(
            param_list,
            lr=self.lr,
            betas=self.betas,
        )
        scheduler = getattr(lr_scheduler, self.lr_scheduler)(
            optimizer, **self.lr_scheduler_args
        )
        return [optimizer], [dict(scheduler=scheduler, interval="step", frequency=1)]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embedding_dim", type=int, default=256)
        parser.add_argument("--n_codes", type=int, default=2048)
        parser.add_argument("--n_hiddens", type=int, default=240)
        parser.add_argument("--n_res_layers", type=int, default=4)
        parser.add_argument("--downsample", nargs="+", type=int, default=(4, 4, 4))
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
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2), **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3), **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4), **kwargs)

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
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = embeddings.movedim(-1, 1)

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

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
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return dict(
            embeddings=embeddings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
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
            conv = SamePadConv3d(in_channels, n_hiddens, kernel_size=4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
            in_channels = n_hiddens
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)
        self.use_attn = use_attn
        if use_attn:
            self.res_stack = nn.Sequential(
                *[AttentionResidualBlock(n_hiddens) for _ in range(n_res_layers)],
                nn.BatchNorm3d(n_hiddens),
                nn.ReLU()
            )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
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
                *[AttentionResidualBlock(n_hiddens) for _ in range(n_res_layers)],
                nn.BatchNorm3d(n_hiddens),
                nn.ReLU()
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
                h = F.relu(h)
        return h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
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
        return self.convt(F.pad(x, self.pad_input))


class LowRankEncoder(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=False):
        super().__init__()
        self.A = nn.Linear(in_features=in_features, out_features=rank, bias=bias)
        self.B = nn.Linear(in_features=rank, out_features=out_features, bias=bias)

    def forward(self, x):
        return self.B(self.A(x))