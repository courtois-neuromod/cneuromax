# Adapted from https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/gpt.py
import itertools
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl

from src.videogpt.resnet import resnet34, ResidualBlock
from src.videogpt.attention import AttentionStack, LayerNorm, AddBroadcastPosEmbed

from src.videogpt.vqvae import VQVAE
from src.videogpt.vqvae_ba import VQVAE as VQVAE_BA


class VideoGPT(pl.LightningModule):
    def __init__(self, args, vqvae_ckpt=None, desc=None):
        # TODO: add docstring describing args
        super().__init__()
        self.args = args
        self.desc = desc

        # Load VQ-VAE and set all parameters to no grad
        if vqvae_ckpt is None:
            vqvae_ckpt = args.vqvae
        try:
            self.vqvae = VQVAE.load_from_checkpoint(vqvae_ckpt)
        except RuntimeError:
            self.vqvae = VQVAE_BA.load_from_checkpoint(vqvae_ckpt)

        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # TODO: condition on actions in addition to embeddings

        # ResNet34 for frame conditioning
        self.use_frame_cond = args.n_cond_frames > 0
        if self.use_frame_cond:
            cond_shape = (
                args.n_cond_frames,
                args.resolution // 4,
                args.resolution // 4,
                240,
            )
            self.resnet = resnet34(1, (1, 4, 4), resnet_dim=240)
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=cond_shape[:-1], embd_dim=cond_shape[-1]
            )

        # ResNet for embedding conditioning
        self.use_emb_cond = hasattr(args, "emb_cond") and args.emb_cond
        if self.use_emb_cond:
            cond_shape = (*self.vqvae.latent_shape, 240)
            self.resnet = ResidualBlock(
                self.vqvae.embedding_dim, 240, stride=1, use_projection=True
            )
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=cond_shape[:-1], embd_dim=cond_shape[-1]
            )

        self.use_mult_emb_cond = hasattr(args, "mult_emb_cond") and args.mult_emb_cond
        if self.use_mult_emb_cond:
            cond_shape = (
                args.n_chunk_cond,
                *self.vqvae.latent_shape,
                self.vqvae.embedding_dim,
            )
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=cond_shape[:-1], embd_dim=cond_shape[-1]
            )
            self.n_chunk_cond = args.n_chunk_cond

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
            args.class_cond_dim,
            cond_shape,
        )

        self.norm = LayerNorm(args.hidden_dim, args.class_cond_dim)

        self.fc_out = nn.Linear(args.hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, args.hidden_dim))

        # Predict reward, actions or done
        self.predict_reward = (
            args.predict_reward if hasattr(args, "predict_reward") else False
        )
        self.predict_actions = (
            args.predict_actions if hasattr(args, "predict_actions") else False
        )
        self.predict_done = (
            args.predict_done if hasattr(args, "predict_done") else False
        )

        flatten_latent_shape = args.hidden_dim * np.prod(self.shape)
        self.flatten_latent_shape = flatten_latent_shape
        if self.predict_reward:
            self.fc_reward = nn.Linear(flatten_latent_shape, self.vqvae.sequence_length)
            self.reward_loss_weight = args.reward_loss_weight
        if self.predict_done:
            self.fc_done = nn.Linear(flatten_latent_shape, self.vqvae.sequence_length)
            self.done_loss_weight = args.done_loss_weight
        if self.predict_actions:
            self.fc_actions = nn.Linear(
                flatten_latent_shape, args.n_actions * self.vqvae.sequence_length
            )
            self.actions_loss_weight = args.actions_loss_weight

        # caches for faster decoding (if necessary)
        self.cond_cache = None

        self.save_hyperparameters()

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def sample(self, n, batch=None):
        device = self.fc_in.weight.device

        cond = dict()
        if self.use_frame_cond or self.args.class_cond or self.use_emb_cond:
            assert batch is not None
            video = batch

            if self.args.class_cond:
                label = batch[1]
                cond["class_cond"] = F.one_hot(label, self.args.class_cond_dim).type_as(
                    video
                )
            if self.use_frame_cond:
                cond["frame_cond"] = video[:, :, : self.args.n_cond_frames]
            if self.use_emb_cond:
                cond["emb_cond"] = video

        samples = torch.zeros((n,) + self.shape).long().to(device)
        h = torch.zeros((n,) + self.shape + (self.hidden_dim,))
        idxs = list(itertools.product(*[range(s) for s in self.shape]))
        targets = {}

        with torch.no_grad():
            prev_idx = None
            for i, idx in enumerate(tqdm(idxs)):
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

                targets["latents"] = samples_slice
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
            other_outputs = tuple()
            if self.predict_actions:
                actions = self.fc_actions(torch.flatten(h, start_dim=1))
                other_outputs += (actions,)
            if self.predict_reward:
                reward = self.fc_reward(torch.flatten(h, start_dim=1))
                other_outputs += (reward,)
            if self.predict_done:
                done = self.fc_done(torch.flatten(h, start_dim=1))
                other_outputs += (done,)

        outputs = samples if not other_outputs else (samples,) + other_outputs
        return outputs

    def forward(self, x, targets, cond, decode_step=None, decode_idx=None):
        if self.use_frame_cond:
            if decode_step is None:
                cond["frame_cond"] = self.cond_pos_embd(self.resnet(cond["frame_cond"]))
            elif decode_step == 0:
                self.cond_cache = self.cond_pos_embd(self.resnet(cond["frame_cond"]))
                cond["frame_cond"] = self.cond_cache
            else:
                cond["frame_cond"] = self.cond_cache

        elif self.use_emb_cond:
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

        elif self.use_mult_emb_cond:
            if decode_step is None or decode_step == 0:
                embs = cond["emb_cond"]
                batch_size = embs.shape[0]
                embs = embs.reshape((-1, *embs.shape[2:]))
                with torch.no_grad():
                    embs = self.vqvae.encode(embs, include_embeddings=True)[1]
                embs = embs.reshape((batch_size, self.n_chunk_cond, *embs.shape[1:]))
                embs = self.cond_pos_embd(embs.movedim(-4, -1))
                cond["emb_cond"] = embs
            if decode_step == 0:
                self.cond_cache = cond["emb_cond"]
            elif decode_step is not None:
                cond["emb_cond"] = self.cond_cache

        h = self.fc_in(x)
        h = self.attn_stack(h, cond, decode_step, decode_idx)
        h = self.norm(h, cond)
        logits = self.fc_out(h)

        if decode_step is None:
            loss = F.cross_entropy(logits.movedim(-1, 1), targets["latents"])
            reward, done, actions = None, None, None

            if self.predict_reward:
                reward = self.fc_reward(torch.flatten(h, start_dim=1))
                loss += self.reward_loss_weight * F.mse_loss(reward, targets["reward"])
            if self.predict_done:
                done = self.fc_done(torch.flatten(h, start_dim=1))
                loss += self.done_loss_weight * F.cross_entropy(done, targets["done"])
            if self.predict_actions:
                actions = self.fc_actions(torch.flatten(h, start_dim=1))
                loss += self.actions_loss_weight * F.cross_entropy(
                    actions, targets["actions"]
                )
            return loss
        else:
            return logits, h

    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch["frames"] if type(batch) == dict else batch

        cond = dict()
        if self.args.class_cond:
            label = batch["class"]
            cond["class_cond"] = F.one_hot(label, self.args.class_cond_dim).type_as(x)
        if self.use_frame_cond:
            cond["frame_cond"] = x[:, :, : self.args.n_cond_frames]
        if self.use_emb_cond:
            cond["emb_cond"] = x[:, :, : self.vqvae.sequence_length]
            x = x[:, :, -self.vqvae.sequence_length :]
        if self.use_mult_emb_cond:
            cond["emb_cond"] = x[:, :-1]
            x = x[:, -1]

        targets = {}
        with torch.no_grad():
            targets["latents"], x = self.vqvae.encode(x, include_embeddings=True)
            # target latents = encodings, x = embeddings
            x = x.movedim(1, -1)

        if self.predict_reward:
            targets["reward"] = batch["reward"]
        if self.predict_done:
            targets["done"] = batch["done"]
        if self.predict_actions:
            targets["actions"] = batch["actions"]

        loss = self(x, targets, cond)
        self.log("tng/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.lr, betas=self.args.betas
        )
        scheduler = getattr(lr_scheduler, self.args.lr_scheduler)(
            optimizer, **self.args.lr_scheduler_args
        )
        return [optimizer], [dict(scheduler=scheduler, interval="step", frequency=1)]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--vqvae", type=str, help="path to vqvae ckpt")
        parser.add_argument("--n_cond_frames", type=int, default=0)
        parser.add_argument("--class_cond", action="store_true")
        parser.add_argument("--emb_cond", action="store_true")
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
