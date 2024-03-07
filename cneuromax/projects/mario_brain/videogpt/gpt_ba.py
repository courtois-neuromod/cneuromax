# Adapted from https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/gpt.py
import itertools
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl

from src.videogpt.resnet import ResidualBlock
from src.videogpt.attention import AttentionStack, LayerNorm, AddBroadcastPosEmbed
from src.videogpt.vqvae_ba import LowRankEncoder


class VideoGPT(pl.LightningModule):
    def __init__(self, args, vqvae_ckpt=None, desc=None):
        # TODO: add docstring describing args
        super().__init__()
        self.args = args
        self.desc = desc

        # Load VQ-VAE and set all parameters to no grad
        from src.videogpt.vqvae_ba import VQVAE

        vqvae_ckpt = args.vqvae if vqvae_ckpt is None else vqvae_ckpt

        try:
            self.vqvae = VQVAE.load_from_checkpoint(vqvae_ckpt)
        except TypeError:  # TODO: cleaner way to detect if vqvae ba or no ba
            from src.videogpt.vqvae import VQVAE as VQVAENOBA

            self.vqvae = VQVAENOBA.load_from_checkpoint(vqvae_ckpt)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # ResNet for embedding conditioning
        cond_shape = (*self.vqvae.latent_shape, 240)
        self.resnet = ResidualBlock(
            self.vqvae.embedding_dim, 240, stride=1, use_projection=True
        )
        self.cond_pos_embd = AddBroadcastPosEmbed(
            shape=cond_shape[:-1], embd_dim=cond_shape[-1]
        )

        # VideoGPT transformer
        self.shape = self.vqvae.latent_shape  # dowsampled (t, h, w)
        self.hidden_dim = args.hidden_dim

        self.fc_in = nn.Linear(self.vqvae.embedding_dim, args.hidden_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        self.attn_stack = AttentionStack(
            self.shape,
            args.hidden_dim,
            args.heads,
            args.layers,
            args.dropout,
            args.attn_type,
            args.attn_dropout,
            None,
            cond_shape,
        )

        self.norm = LayerNorm(args.hidden_dim, class_cond_dim=None)

        self.fc_out = nn.Linear(args.hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, args.hidden_dim))

        # caches for faster decoding (if necessary)
        self.cond_cache = None

        # Brain encoding
        flatten_latent_shape = args.hidden_dim * np.prod(self.shape)
        self.flatten_latent_shape = flatten_latent_shape
        if not hasattr(args, "encoder_rank") or args.encoder_rank is None:
            self.brain_encoding = nn.Linear(
                in_features=flatten_latent_shape, out_features=args.n_voxels
            )
        else:
            self.brain_encoding = LowRankEncoder(
                in_features=flatten_latent_shape,
                out_features=args.n_voxels,
                rank=args.encoder_rank,
            )
        if args.frozen_encoding:
            for param in self.brain_encoding.parameters():
                param.requires_grad = False
        self.frozen_encoding = args.frozen_encoding
        self.brain_loss_lambda = args.brain_loss_lambda
        self.n_voxels = args.n_voxels
        self.weight_decay = args.weight_decay
        self.bal = args.bal if hasattr(args, "bal") else args.brain_loss_lambda > 0

        # regiter hooks to get activations for brain encoding
        self.activations = None
        self.regiter_activations = False
        self.val_autoreg_loss = None

        def make_hook(name):
            def hook(model, input, output):
                if self.register_activations:
                    self.activations = output

            return hook

        found_layer = False
        for name, module in self.named_modules():
            if name == args.layer_name:
                module.register_forward_hook(make_hook(name))
                found_layer = True
                break
        if not found_layer:
            raise RuntimeError(
                "Couldn't attach hook to the layer,"
                f"{args.layer_name} not found in model."
            )

        self.save_hyperparameters()
        self.batch_idx_slice = None
        self.val_autoreg_loss_of_epoch = []

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def sample(self, n, batch=None):
        device = self.fc_in.weight.device

        cond = dict()
        cond["emb_cond"] = batch

        samples = torch.zeros((n,) + self.shape).long().to(device)
        h = torch.zeros((n,) + self.shape + (self.hidden_dim,))
        idxs = list(itertools.product(*[range(s) for s in self.shape]))
        targets = {}

        with torch.no_grad():
            self.register_activations = False
            prev_idx = None
            for i, idx in enumerate(idxs):
                batch_idx_slice = (slice(None, None), *[slice(i, i + 1) for i in idx])
                batch_idx = (slice(None, None), *idx)
                embeddings = self.vqvae.codebook.dictionary_lookup(samples)

                if prev_idx is None:
                    # set arbitrary input values for the first token
                    # does not matter what value since it will be shifted anyways
                    embeddings_slice = embeddings[batch_idx_slice]
                    samples_slice = samples[batch_idx_slice]
                else:
                    embeddings_slice = embeddings[prev_idx]
                    samples_slice = samples[prev_idx]

                targets["codes"] = samples_slice
                self.batch_idx_slice = batch_idx_slice
                logits, h_slice = self(
                    embeddings_slice, targets, cond, decode_step=i, decode_idx=idx
                )

                # squeeze all possible dim except batch dimension
                logits = (
                    logits.squeeze().unsqueeze(0)
                    if logits.shape[0] == 1
                    else logits.squeeze()
                )
                probs = F.softmax(logits, dim=-1)
                samples[batch_idx] = torch.multinomial(probs, 1).squeeze(-1)
                h[batch_idx_slice] = h_slice

                prev_idx = batch_idx_slice
            embeddings = self.vqvae.codebook.dictionary_lookup(samples)
            samples = self.vqvae.decode(samples)
            samples = torch.clamp(samples, -0.5, 0.5)  # BCTHW in [-0.5, 0.5]
        self.batch_idx_slice = None
        return samples

    def forward(self, x, targets, cond, decode_step=None, decode_idx=None):
        if decode_step is None:
            with torch.no_grad():
                cond["emb_cond"] = self.vqvae.encode(
                    cond["emb_cond"], include_embeddings=True
                )[1]
            cond["emb_cond"] = self.cond_pos_embd(
                self.resnet(cond["emb_cond"]).movedim(-4, -1)
            )
        elif decode_step == 0:
            with torch.no_grad():
                cond["emb_cond"] = self.vqvae.encode(
                    cond["emb_cond"], include_embeddings=True
                )[1]
            self.cond_cache = self.cond_pos_embd(
                self.resnet(cond["emb_cond"]).movedim(-4, -1)
            )
            cond["emb_cond"] = self.cond_cache
        else:
            cond["emb_cond"] = self.cond_cache

        h = self.fc_in(x)
        h = self.attn_stack(h, cond, decode_step, decode_idx)
        h = self.norm(h, cond)
        logits = self.fc_out(h)

        if decode_step is None:
            loss = F.cross_entropy(logits.movedim(-1, 1), targets["codes"])
            return loss
        else:
            return logits, h

    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch["frames"]

        cond = dict()
        cond["emb_cond"] = x[:, :-1].reshape(
            (x.shape[0] * (x.shape[1] - 1), *x.shape[2:])
        )

        targets = dict()
        target_frames = x[:, 1:].reshape((x.shape[0] * (x.shape[1] - 1), *x.shape[2:]))
        with torch.no_grad():
            targets["codes"], target_emb = self.vqvae.encode(
                target_frames, include_embeddings=True
            )
        target_emb = target_emb.movedim(1, -1)

        self.register_activations = True
        autoreg_loss = self(target_emb, targets, cond)
        self.register_activations = False

        loss_lambda = self.brain_loss_lambda
        activations = self.activations
        if not self.bal:
            activations = activations.clone().detach()
        activations = activations.reshape(
            (x.shape[0], x.shape[1] - 1, *self.activations.shape[1:])
        )
        activations = activations.mean(dim=1).flatten(start_dim=1)
        brain_encoding = self.brain_encoding(activations)
        brain_loss = F.mse_loss(brain_encoding, batch["bold"][:, : self.n_voxels])
        self.log("tng/autoreg_loss", autoreg_loss, prog_bar=True)
        self.log("tng/brain_loss", brain_loss)
        loss = (1 - loss_lambda) * autoreg_loss + loss_lambda * brain_loss
        self.log("tng/total_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["frames"] if isinstance(batch, dict) else batch

        cond = dict()
        cond["emb_cond"] = x[:, :-1].reshape(
            (x.shape[0] * (x.shape[1] - 1), *x.shape[2:])
        )

        targets = dict()
        target_frames = x[:, 1:].reshape((x.shape[0] * (x.shape[1] - 1), *x.shape[2:]))
        targets["codes"], target_emb = self.vqvae.encode(
            target_frames, include_embeddings=True
        )
        target_emb = target_emb.movedim(1, -1)

        self.register_activations = True
        autoreg_loss = self(target_emb, targets, cond)
        self.register_activations = False

        activations = self.activations
        loss_lambda = 1 if self.brain_loss_lambda == 0 else self.brain_loss_lambda
        activations = activations.reshape(
            (x.shape[0], x.shape[1] - 1, *self.activations.shape[1:])
        )
        activations = activations.mean(dim=1).flatten(start_dim=1)
        brain_encoding = self.brain_encoding(activations)
        brain_loss = F.mse_loss(brain_encoding, batch["bold"][:, : self.n_voxels])
        self.log("val/autoreg_loss", autoreg_loss, prog_bar=True, sync_dist=True)
        self.log("val/brain_loss", brain_loss, sync_dist=True)
        loss = (1 - self.brain_loss_lambda) * autoreg_loss + loss_lambda * brain_loss
        self.log("val/total_loss", loss, sync_dist=True)
        self.val_autoreg_loss_of_epoch.append(autoreg_loss.item())

        return {
            "autoreg_loss": autoreg_loss,
            "brain_loss": brain_loss,
            "total_loss": loss,
        }

    def on_validation_epoch_end(self):
        mean_val_autoreg_loss = np.mean(self.val_autoreg_loss_of_epoch)
        if self.best_val_autoreg_loss is None:
            self.best_val_autoreg_loss = mean_val_autoreg_loss
        else:
            self.best_val_autoreg_loss = min(
                self.best_val_autoreg_loss, mean_val_autoreg_loss
            )
        self.val_autoreg_loss_of_epoch.clear()

    def configure_optimizers(self):
        param_list = [
            {"params": self.fc_in.parameters()},
            {"params": self.resnet.parameters()},
            {"params": self.cond_pos_embd.parameters()},
            {"params": self.attn_stack.parameters()},
            {"params": self.norm.parameters()},
            {"params": self.fc_out.parameters()},
        ]
        if not self.frozen_encoding:
            param_list.append(
                {
                    "params": self.brain_encoding.parameters(),
                    "weight_decay": self.weight_decay,
                }
            )
        optimizer = torch.optim.Adam(
            param_list,
            lr=self.args.lr,
            betas=self.args.betas,
        )
        scheduler = getattr(lr_scheduler, self.args.lr_scheduler)(
            optimizer, **self.args.lr_scheduler_args
        )
        return [optimizer], [dict(scheduler=scheduler, interval="step", frequency=1)]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--vqvae", type=str, help="path to vqvae ckpt")
        parser.add_argument("--predict_reward", action="store_true")
        parser.add_argument("--reward_loss_weight", type=float, default=1.0)
        parser.add_argument("--predict_done", action="store_true")
        parser.add_argument("--done_loss_weight", type=float, default=1.0)
        parser.add_argument("--predict_actions", action="store_true")
        parser.add_argument("--actions_loss_weight", type=float, default=1.0)

        # VideoGPT hyperparmeters
        parser.add_argument("--hidden_dim", type=int, default=576)
        parser.add_argument("--heads", type=int, default=4)
        parser.add_argument("--layers", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument(
            "--attn_type", type=str, default="full", choices=["full", "sparse"]
        )
        parser.add_argument("--attn_dropout", type=float, default=0.3)

        return parser
