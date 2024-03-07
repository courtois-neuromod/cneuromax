import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from src.videogpt.vqvae_ba import LowRankEncoder


class VideoGPTEncoder(pl.LightningModule):
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

        # VideoGPT transformer
        self.shape = self.vqvae.latent_shape  # dowsampled (t, h, w)
        self.hidden_dim = args.hidden_dim

        # Maybe remove fc_in
        self.fc_in = nn.Linear(self.vqvae.embedding_dim, args.hidden_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim,
            nhead=args.heads,
            dim_feedforward=args.hidden_dim,
            dropout=args.dropout,
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=args.layers
        )

        self.pos_emb = PositionalEncoding(self.shape, args.hidden_dim, args.dropout)

        self.fc_out = nn.Linear(args.hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, args.hidden_dim))

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
        self.register_activations = False
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

        self.val_autoreg_loss_of_epoch = []
        self.save_hyperparameters()

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def sample(self, batch, n=1):
        device = self.fc_in.weight.device

        batch = batch.to(device)
        if len(batch.shape) == 4:  # no batch dimension
            batch = torch.stack([batch] * n, dim=0)
        elif batch.shape[0] == 1:
            batch = torch.stack([batch[0]] * n, dim=0)
        bs = batch.shape[0]

        with torch.no_grad():
            logits = self(batch)
            probs = F.softmax(logits, dim=-1).view(-1, self.vqvae.n_codes)
            samples = torch.multinomial(probs, 1).squeeze(-1)
            samples = self.vqvae.decode(samples.view(bs, *self.shape))
            samples = torch.clamp(samples, -0.5, 0.5)
        return samples

    def forward(self, x):
        h = self.vqvae.encode(x, include_embeddings=True)[1].movedim(-4, -1)
        h = self.fc_in(h)
        h = self.pos_emb(h)
        h = h.view(h.shape[0], np.prod(self.shape), self.hidden_dim)
        h = self.transformer(h)
        logits = self.fc_out(h)
        logits = logits.view(h.shape[0], *self.shape, self.vqvae.n_codes)
        return logits

    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch["frames"]

        input_frames = x[:, :-1].reshape((x.shape[0] * (x.shape[1] - 1), *x.shape[2:]))
        target_frames = x[:, 1:].reshape((x.shape[0] * (x.shape[1] - 1), *x.shape[2:]))
        with torch.no_grad():
            target_codes = self.vqvae.encode(target_frames, include_embeddings=False)

        self.register_activations = True
        logits = self(input_frames)
        self.register_activations = False
        autoreg_loss = F.cross_entropy(logits.movedim(-1, 1), target_codes)

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
        x = batch["frames"]

        input_frames = x[:, :-1].reshape((x.shape[0] * (x.shape[1] - 1), *x.shape[2:]))

        target_frames = x[:, 1:].reshape((x.shape[0] * (x.shape[1] - 1), *x.shape[2:]))
        target_codes = self.vqvae.encode(target_frames, include_embeddings=False)

        self.register_activations = True
        logits = self(input_frames)
        self.register_activations = False
        autoreg_loss = F.cross_entropy(logits.movedim(-1, 1), target_codes)

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
            {"params": self.pos_embd.parameters()},
            {"params": self.transformer.parameters()},
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

        # VideoGPT hyperparmeters
        parser.add_argument("--hidden_dim", type=int, default=576)
        parser.add_argument("--heads", type=int, default=4)
        parser.add_argument("--layers", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.2)

        return parser


class PositionalEncoding(nn.Module):
    def __init__(self, shape, emb_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.shape = shape
        self.emb_dim = emb_dim
        self.n_dim = n_dim = len(shape)
        if not emb_dim % n_dim:
            emb_dim_sizes = [emb_dim // n_dim] * n_dim
        else:
            emb_dim_sizes = [emb_dim // n_dim - (n_dim - 1 - emb_dim % n_dim)]
            emb_dim_sizes += [emb_dim // n_dim + 1] * (n_dim - 1)
        self.emb = nn.ParameterDict(
            {
                f"d_{i}": nn.Parameter(torch.randn(shape[i], emb_dim_sizes[i]) * 0.01)
                for i in range(n_dim)
            }
        )
        self.emb_dim_sizes = emb_dim_sizes

    def forward(self, x):
        # To verify
        embs = []
        for i in range(self.n_dim):
            e = self.emb[f"d_{i}"]
            e = e.view(
                1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1
            )
            e = e.expand(1, *self.shape, -1)
            embs.append(e)
        embs = torch.cat(embs, dim=-1)
        x = x + embs
        return self.dropout(x)
