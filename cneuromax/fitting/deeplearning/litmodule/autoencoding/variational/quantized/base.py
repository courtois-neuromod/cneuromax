import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from vector_quantize_pytorch import VectorQuantize


class BaseVQVAEModel(BaseVAEModel):
    def __init__(
        self,
        encoder_model: nn.Sequential,
        decoder_model: nn.Sequential,
    ) -> None:

        super().__init__()

        self.encoder = Encoder(encoder_model)
        self.decoder = Decoder(decoder_model)

        self.latent_shape = encoder_sequential[-1].shape[1:]

    def forward(self, x):
        """
        Args:
            x         batch_size x (x_shape)
        Returns:
            recon_x   batch_size x (x_shape)
            mu        batch_size x (latent_shape)
            log_sigma batch_size x (latent_shape)
        """
        mu, log_sigma = self.encoder(x)

        sigma = torch.exp(log_sigma)
        epsilon = torch.randn(self.latent_shape, device=self.device)

        z = mu + sigma * epsilon

        recon_x = self.decoder(z)

        return recon_x, mu, log_sigma

    def step(self, batch):
        x = batch
        recon_x, mu, log_sigma = self(x)
        loss = compute_loss(x, recon_x, mu, log_sigma)

        return loss

    def training_step(self, batch, _):
        loss = self.step(batch)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, _):
        loss = self.step(batch)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters())


def compute_loss(x, recon_x, mu, log_sigma):
    """
    Args:
        x         batch_size x (x_shape)
        recon_x   batch_size x (x_shape)
        mu        batch_size x (latent_shape)
        log_sigma batch_size x (latent_shape)
    """

    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    kl_loss = -0.5 * torch.sum(
        1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
    )

    return recon_loss + kl_loss


class Encoder(nn.Module):
    def __init__(self, model: nn.Sequential) -> None:
        super().__init__()

        self.model = model

        self.latent_shape = self.model[-1].shape[1:]

    def forward(self, x):
        """
        Args:
            x         batch_size x (x_size)
        Returns:
            mu        batch_size x (latent_size)
            log_sigma batch_size x (latent_size)
        """
        mu_log_sigma = self.model(x)
        mu, log_sigma = mu_log_sigma.split(self.latent_shape, dim=-1)

        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, model: nn.Sequential) -> None:
        super().__init__()

        self.model = model

    def forward(self, z):
        """
        Args:
            z         batch_size x (latent_shape)
        Returns:
            recon_x   batch_size x (x_shape)
        """
        recon_x = self.model(z)

        return recon_x
