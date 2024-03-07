import pytorch_lightning as pl
import torch
import math
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from src.videogpt.gpt_ba import VideoGPT


class ActionModule(pl.LightningModule):
    def __init__(
        self,
        gpt_layer_name,
        n_actions,
        gpt_ckpt,
        vqvae_ckpt,
        lr=1e-4,
        betas=(0.9, 0.999),
        lr_scheduler="CosineAnnealingLR",
        lr_scheduler_args={"T_max": 50000},
        pos_weight=None,  # should be n_negatives / n_positives for each class
        weight_decay=0,
        fast_training=False,
    ):
        super().__init__()
        self.gpt_layer_name = gpt_layer_name
        self.n_actions = n_actions
        self.gpt_ckpt = gpt_ckpt
        self.vqvae_ckpt = vqvae_ckpt
        self.lr = lr
        self.betas = betas
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args
        self.weight_decay = weight_decay

        dummy_param = nn.Parameter(torch.empty(0))
        self.gpt = VideoGPT.load_from_checkpoint(
            gpt_ckpt, vqvae_ckpt=vqvae_ckpt, map_location=dummy_param.device
        ).eval()
        for p in self.gpt.parameters():
            p.requires_grad = False
        n_activations = self.gpt.flatten_latent_shape
        self.action_readout = nn.Linear(n_activations, n_actions)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_val = nn.BCELoss()

        self.register_activations = False

        def make_hook(name):
            def hook(model, input, output):
                if self.register_activations:
                    layer_activations = output
                    self.activations[
                        self.gpt.batch_idx_slice
                    ] = layer_activations.detach().clone()

            return hook

        found_layer = False
        for name, module in self.gpt.named_modules():
            if name == gpt_layer_name:
                module.register_forward_hook(make_hook(name))
                found_layer = True
                break
        if not found_layer:
            raise RuntimeError(
                "Couldn't attach hook to the layer, "
                f"{gpt_layer_name} not found in model."
            )
        self.fast_trainnig = fast_training
        if fast_training:
            self.training_step = self.training_step_fast
        else:
            self.training_step = self.training_step_slow
        self.save_hyperparameters()

    def forward(self, x):
        bs = x.shape[0]
        self.activations = torch.zeros(
            (bs, *self.gpt.shape, self.gpt.hidden_dim), device=self.device
        )
        self.register_activations = True
        gpt_pred = self.gpt.sample(bs, x)
        self.register_activations = False
        actions = self.action_readout(self.activations.flatten(start_dim=1))
        if not self.training:
            actions = torch.sigmoid(actions)
            # During training the Sigmoid is done in the BCEwithLogits loss
        return actions, gpt_pred

    def training_step_slow(self, batch, batch_idx):
        x = batch["frames"]
        y = batch["target_actions"]
        actions, _ = self(x)
        loss = self.bce_loss(actions, y)
        self.log("tng/loss", loss, prog_bar=True)
        return loss

    def training_step_fast(self, batch, batch_idx):
        x = batch["frames"]
        y = batch["target_actions"]

        cond = {"emb_cond": x[:, :, : self.gpt.vqvae.sequence_length]}
        target_frames = x[:, :, self.gpt.vqvae.sequence_length :]
        targets = dict()
        with torch.no_grad():
            targets["codes"], target_emb = self.gpt.vqvae.encode(
                target_frames, include_embeddings=True
            )
        target_emb = target_emb.movedim(1, -1)
        self.gpt.register_activations = True
        _ = self.gpt(target_emb, targets, cond)
        self.gpt.register_activations = False
        activations = self.gpt.activations.clone().detach()

        actions = self.action_readout(activations.flatten(start_dim=1))
        loss = self.bce_loss(actions, y)
        self.log("tng/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["frames"][:, :, : self.gpt.vqvae.sequence_length]
        y = batch["target_actions"]
        actions, _ = self(x)
        loss = self.bce_val(actions, y)
        actions_thres05 = actions > 0.5

        precision_thres05 = (
            ((actions_thres05 + y) == 2).sum() / y.sum() if y.sum() else 0.5
        )
        recall_thres05 = (
            ((actions_thres05 + y) == 2).sum() / actions_thres05.sum()
            if actions_thres05.sum()
            else 0.5
        )
        accuracy_thres05 = (actions_thres05 == y).mean(dtype=torch.float)
        actions_thres08 = actions > 0.8
        precision_thres08 = (
            ((actions_thres08 + y) == 2).sum() / y.sum() if y.sum() else 0.5
        )
        recall_thres08 = (
            ((actions_thres08 + y) == 2).sum() / actions_thres08.sum()
            if actions_thres08.sum()
            else 0.5
        )
        accuracy_thres08 = (actions_thres08 == y).mean(dtype=torch.float)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/accuracy_thres=0.5", accuracy_thres05, sync_dist=True)
        self.log("val/accuracy_thres=0.8", accuracy_thres08, sync_dist=True)
        self.log("val/precision_thres=0.5", precision_thres05, sync_dist=True)
        self.log("val/precision_thres=0.8", precision_thres08, sync_dist=True)
        self.log("val/recall_thres=0.5", recall_thres05, sync_dist=True)
        self.log("val/recall_thres=0.8", recall_thres08, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.action_readout.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        scheduler = getattr(lr_scheduler, self.lr_scheduler)(
            optimizer, **self.lr_scheduler_args
        )
        return [optimizer], [dict(scheduler=scheduler, interval="step", frequency=1)]


class TransformerActionModule(pl.LightningModule):
    # Maybe add positional encoding to GPT activations
    def __init__(
        self,
        nhead,
        num_layers,
        gpt_layer_name,
        n_buttons,
        n_actions_context,
        gpt_ckpt,
        vqvae_ckpt,
        dim_feedforward=2048,
        dropout=0.1,
        lr=1e-4,
        betas=(0.9, 0.999),
        lr_scheduler="CosineAnnealingLR",
        lr_scheduler_args={"T_max": 50000},
        pos_weight=None,  # should be n_negatives / n_positives for each class
        weight_decay=0,
        fast_training=False,
    ):
        super().__init__()
        self.gpt_layer_name = gpt_layer_name
        self.n_buttons = n_buttons
        self.n_actions_context = n_actions_context
        self.gpt_ckpt = gpt_ckpt
        self.vqvae_ckpt = vqvae_ckpt
        self.lr = lr
        self.betas = betas
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args
        self.weight_decay = weight_decay
        self.num_layers = num_layers

        self.gpt = VideoGPT.load_from_checkpoint(gpt_ckpt, vqvae_ckpt=vqvae_ckpt).eval()
        for p in self.gpt.parameters():
            p.requires_grad = False

        if self.num_layers:
            # activations shape  (bs, *self.gpt.shape, self.gpt.hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.gpt.hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=dropout,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.positional_encoding = PositionalEncoding(self.gpt.hidden_dim)

        if n_actions_context:
            self.n_embeddings = 2**n_buttons
            self.action_embeddings = nn.Embedding(
                self.n_embeddings, self.gpt.hidden_dim
            )
        self.fc_out = nn.Linear(
            self.gpt.hidden_dim * (n_actions_context + np.prod(self.gpt.shape)),
            n_buttons,
        )

        self.gpt = self.gpt.to(next(self.fc_out.parameters()).device)
        if self.num_layers:
            self.positional_encoding = self.positional_encoding.to(
                next(self.fc_out.parameters()).device
            )

        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_val = nn.BCELoss()

        self.register_activations = False

        def make_hook(name):
            def hook(model, input, output):
                if self.register_activations:
                    layer_activations = output
                    self.activations[
                        self.gpt.batch_idx_slice
                    ] = layer_activations.detach().clone()

            return hook

        found_layer = False
        for name, module in self.gpt.named_modules():
            if name == gpt_layer_name:
                module.register_forward_hook(make_hook(name))
                found_layer = True
                break
        if not found_layer:
            raise RuntimeError(
                "Couldn't attach hook to the layer, "
                f"{gpt_layer_name} not found in model."
            )
        self.fast_trainnig = fast_training
        if fast_training:
            self.training_step = self.training_step_fast
        else:
            self.training_step = self.training_step_slow
        self.save_hyperparameters()

    def forward(self, frames, actions):
        bs = frames.shape[0]
        self.activations = torch.zeros(
            (bs, *self.gpt.shape, self.gpt.hidden_dim), device=self.device
        )
        self.register_activations = True
        gpt_pred = self.gpt.sample(bs, frames)
        self.register_activations = False
        activations = torch.reshape(self.activations, (bs, -1, self.gpt.hidden_dim))
        if self.n_actions_context:
            actions_ids = (
                actions * (2 ** torch.arange(self.n_buttons, device=actions.device))
            ).sum(axis=-1)
            act_emb = self.action_embeddings(actions_ids)
            x = torch.cat((activations, act_emb), dim=1)
        else:
            x = activations
        if self.num_layers:
            x = self.positional_encoding(x)
            x = self.transformer(x)
        predictions = self.fc_out(x.reshape(bs, -1))
        if not self.training:
            predictions = torch.sigmoid(predictions)
            # During training the Sigmoid is done in the BCEwithLogits loss
        return predictions, gpt_pred

    def training_step_slow(self, batch, batch_idx):
        x_frames = batch["frames"]
        if self.n_actions_context:
            x_actions = batch["context_actions"]
        else:
            x_actions = None
        y = batch["target_actions"]
        actions, _ = self(x_frames, x_actions)
        loss = self.bce_loss(actions, y)
        self.log("tng/loss", loss, prog_bar=True)
        return loss

    def training_step_fast(self, batch, batch_idx):
        x_frames = batch["frames"]
        if self.n_actions_context:
            x_actions = batch["context_actions"]
        else:
            x_actions = None
        y = batch["target_actions"]

        bs = x_frames.shape[0]
        cond = {"emb_cond": x_frames[:, :, : self.gpt.vqvae.sequence_length]}
        target_frames = x_frames[:, :, self.gpt.vqvae.sequence_length :]
        targets = dict()
        with torch.no_grad():
            targets["codes"], target_emb = self.gpt.vqvae.encode(
                target_frames, include_embeddings=True
            )
        target_emb = target_emb.movedim(1, -1)
        self.activations = torch.zeros(
            (bs, *self.gpt.shape, self.gpt.hidden_dim), device=self.device
        )
        self.gpt.register_activations = True
        _ = self.gpt(target_emb, targets, cond)
        self.gpt.register_activations = False
        activations = self.gpt.activations.clone().detach()
        activations = torch.reshape(self.activations, (bs, -1, self.gpt.hidden_dim))
        if self.n_actions_context:
            actions_ids = (
                x_actions * (2 ** torch.arange(self.n_buttons, device=x_actions.device))
            ).sum(axis=-1)
            act_emb = self.action_embeddings(actions_ids)
            x = torch.cat((activations, act_emb), dim=1)
        else:
            x = activations
        if self.num_layers:
            x = self.positional_encoding(x)
            x = self.transformer(x)
        predictions = self.fc_out(x.reshape(bs, -1))
        loss = self.bce_loss(predictions, y)
        self.log("tng/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_frames = batch["frames"][:, :, : self.gpt.vqvae.sequence_length]
        if self.n_actions_context:
            x_actions = batch["context_actions"]
        else:
            x_actions = None
        y = batch["target_actions"]
        actions, _ = self(x_frames, x_actions)
        loss = self.bce_val(actions, y)
        actions_thres05 = actions > 0.5

        precision_thres05 = (
            ((actions_thres05 + y) == 2).sum() / y.sum() if y.sum() else 0.5
        )
        recall_thres05 = (
            ((actions_thres05 + y) == 2).sum() / actions_thres05.sum()
            if actions_thres05.sum()
            else 0.5
        )
        accuracy_thres05 = (actions_thres05 == y).mean(dtype=torch.float)
        actions_thres08 = actions > 0.8
        precision_thres08 = (
            ((actions_thres08 + y) == 2).sum() / y.sum() if y.sum() else 0.5
        )
        recall_thres08 = (
            ((actions_thres08 + y) == 2).sum() / actions_thres08.sum()
            if actions_thres08.sum()
            else 0.5
        )
        accuracy_thres08 = (actions_thres08 == y).mean(dtype=torch.float)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/accuracy_thres=0.5", accuracy_thres05, sync_dist=True)
        self.log("val/accuracy_thres=0.8", accuracy_thres08, sync_dist=True)
        self.log("val/precision_thres=0.5", precision_thres05, sync_dist=True)
        self.log("val/precision_thres=0.8", precision_thres08, sync_dist=True)
        self.log("val/recall_thres=0.5", recall_thres05, sync_dist=True)
        self.log("val/recall_thres=0.8", recall_thres08, sync_dist=True)

    def configure_optimizers(self):
        param_list = [{"params": self.fc_out.parameters()}]
        if self.n_actions_context:
            param_list.append({"params": self.context_embeddings.parameters()})
        if self.num_layers:
            param_list.append({"params": self.transformer.parameters()})

        optimizer = torch.optim.Adam(
            param_list,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        scheduler = getattr(lr_scheduler, self.lr_scheduler)(
            optimizer, **self.lr_scheduler_args
        )
        return [optimizer], [dict(scheduler=scheduler, interval="step", frequency=1)]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
