import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.utils.rnn import pack_padded_sequence
from torchscript import TensorType

from cntrain.dl.common.base.model import BaseModel


class Ha2018MDNRNN(BaseModel):
    """Mixture Density Network RNN (MDN-RNN) as seen in the "World Models" paper
    (Ha & Schmidhuber, 2018) https://arxiv.org/abs/1803.10122.
    """

    def __init__(self, num_features: int, num_actions: int) -> None:
        """Constructor.

        Args:
            num_features: Number of features.
            num_actions: Number of allowed actions.
        """
        assert isinstance(hidden_size, int) and hidden_size >= 1
        assert isinstance(hidden_size, int) and hidden_size >= 1
        assert isinstance(num_gaussians, int) and num_gaussians >= 1
        assert isinstance(predicting_reward, bool)
        assert isinstance(predicting_termination, bool)

        super().__init__()

        self.num_features, self.num_actions = num_features, num_actions

        self.lstm = nn.LSTM(
            input_size=self.num_features + self.num_actions,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_lstm_layers,
            batch_first=True,
        )

        out_features = (
            cfg.num_gaussians  # raw_pi
            + cfg.num_gaussians * self.num_features  # mu
            + cfg.num_gaussians * self.num_features  # log_sigma
            + 1  # rew_hat
            + 1  # done_hat
        )

        self.fc = nn.Linear(cfg.hidden_size, out_features)

    def reset(self, batch_size: int) -> None:
        """Resets the hidden & cell states.

        Args:
            batch_size: Batch Size.
        """
        self.h = torch.zeros(
            cfg.num_lstm_layers, batch_size, cfg.hidden_size
        ).to(self.device)

        self.c = torch.zeros(
            cfg.num_lstm_layers, batch_size, cfg.hidden_size
        ).to(self.device)

    def forward(
        self,
        features_and_actions: TensorType[
            "batch_size", "seq_len", "num_features + num_actions"
        ],
        seq_len: TensorType["batch_size"],
    ) -> tuple(
        TensorType["batch_size * seq_len", "num_gaussians"],
        TensorType["batch_size * seq_len", "num_gaussians", "num_features"],
        TensorType["batch_size * seq_len", "num_gaussians", "num_features"],
        TensorType["batch_size * seq_len"],
        TensorType["batch_size * seq_len"],
    ):
        """Forward step.

        Args:
            features_and_actions: Features & actions concatenated.
            seq_len: Batch-wise sequence lengths.
        Returns:
            log_pi: Gaussian sampling log probabilities.
            mu: Gaussian means.
            sigma: Gaussians standard deviations.
            rew_hat: Reward values predictions.
            done_hat: Episode termination predictions.
        """
        # BS x SL x (NF + NA) -> (BS *~ SL) x (NF + NA)
        x = pack_padded_sequence(
            features_and_actions,
            lengths=seq_len,
            batch_first=True,
            enforce_sorted=False,
        )

        # (BS *~ SL) x (NF + NA) -> (BS * SL) x HS
        x = self.lstm(x)[0][0]

        # (BS * SL) x HS -> (BS * SL) x [NG + (NG * NF) + (NG * NF) + 1 + 1]
        x = self.fc(x)

        raw_pi, mu, log_sigma, rew_hat, done_hat = x.split(
            [
                cfg.num_gaussians,
                cfg.num_gaussians * self.num_features,
                cfg.num_gaussians * self.num_features,
                1,
                1,
            ],
            dim=-1,
        )

        log_pi = F.log_softmax(raw_pi, dim=-1)

        # (BS * SL) x (NG * NF) -> (BS * SL) x NG x NF
        mu = mu.view(x.size(0), cfg.num_gaussians, self.num_features)

        # (BS * SL) x (NG * NF) -> (BS * SL) x NG x NF
        sigma = torch.exp(
            log_sigma.view(
                x.size(0),
                cfg.num_gaussians,
                self.num_features,
            )
        )

        return log_pi, mu, sigma, rew_hat, done_hat

    def predict(
        self,
        features_and_actions: TensorType[
            "batch_size", "num_features + num_actions"
        ],
    ) -> tuple(
        TensorType["batch_size", "num_gaussians"],
        TensorType["batch_size"],
        TensorType["batch_size"],
    ):
        """Batch prediction.

        Args:
            features_and_actions: Features & actions concatenated.
        Returns:
            next_features_hat: Predicted next features.
            rew_hat: Predicted reward signals.
            done_hat: Predicted termination signals.
        """
        # BS x (NF + NA) -> BS x 1 x (NF + NA)
        x = features_and_actions.unsqueeze(1)

        # BS x 1 x (NF + NA) -> BS x 1 x HS
        x, (self.h, self.c) = self.lstm(x, (self.h, self.c))

        # BS x 1 x HS -> BS x 1 x [NG + (NG * NF) + (NG * NF) + 1 + 1]
        x = self.fc(x)

        raw_pi, mu, log_sigma, rew_hat, done_hat = x.split(
            [
                cfg.num_gaussians,
                cfg.num_gaussians * self.num_features,
                cfg.num_gaussians * self.num_features,
                1,
                1,
            ],
            dim=-1,
        )

        log_pi = F.log_softmax(raw_pi, dim=-1)

        # BS x 1 x (NG * NF) -> BS x 1 x NG x NF
        mu = mu.view(x.size(0), 1, cfg.num_gaussians, self.num_features)

        # BS x 1 x (NG * NF) -> BS x 1 x NG x NF
        sigma = torch.exp(
            log_sigma.view(x.size(0), 1, cfg.num_gaussians, self.num_features)
        )

        # BS x 1 x NG -> BS x NG
        pi = torch.exp(log_pi).squeeze()

        gaussian_idx = torch.multinomial(pi, 1)

        # BS x 1 x NG x NF -> BS x NF
        mu = mu[:, :, gaussian_idx, :].squeeze()
        sigma = sigma[:, :, gaussian_idx, :].squeeze()

        next_features_hat = mu + sigma * torch.randn_like(sigma)
        done_hat = done_hat > 0.5

        return next_features_hat, rew_hat, done_hat

    def step(
        self,
        batch: tuple(
            TensorType["batch_size", "seq_len", "num_features + num_actions"],
            TensorType["batch_size", "seq_len", "num_features"],
            TensorType["batch_size", "seq_len"],
            TensorType["batch_size", "seq_len"],
        ),
        stage: str,
    ) -> torch.float:
        """Predict step.

        Args:
            batch:
                features_and_actions: Features & actions concatenated.
                next_features: Next state features.
                rew: Reward signals.
                done: Termination signals.
            stage: Current stage (train/val/test).
        Returns:
            The GMM, BCE & MSE losses combined.
        """
        features_and_actions, next_features, rew, done = batch

        # BS x 1 x NG x NF -> BS x NF
        seq_len = torch.argwhere(done)[:, 1].cpu()

        log_pi, mu, sigma, rew_hat, done_hat = self(
            features_and_actions, seq_len
        )

        gmm_loss = compute_gmm_loss(
            next_features,
            log_pi,
            mu,
            sigma,
            lengths,
        )

        if cfg.predicting_termination:
            bce_loss = F.binary_cross_entropy_with_logits(done_hat, done)
        else:
            bce_loss = 0

        if cfg.predicting_reward:
            mse_loss = F.mse_loss(rew_hat, rew)
        else:
            mse_loss = 0

        self.log(f"{stage}/gmm_loss", gmm_loss)
        self.log(f"{stage}/bce_loss", bce_loss)
        self.log(f"{stage}/mse_loss", mse_loss)

        return gmm_loss + bce_loss + mse_loss


def compute_gmm_loss(
    next_features: TensorType["batch_size", "seq_len", "num_features"],
    log_pi: TensorType["batch_size * seq_len", "num_gaussians"],
    mu: TensorType["batch_size * seq_len", "num_gaussians", "num_features"],
    sigma: TensorType["batch_size * seq_len", "num_gaussians", "num_features"],
    seq_len: TensorType["batch_size"],
) -> torch.float:
    """
    Args:
        next_features: Next state features.
        log_pi: Gaussian sampling log probabilities.
        mu: Gaussian means.
        sigma: Gaussians standard deviations.
    Returns:
        The Gaussian Mixture Model loss.
    """
    # BS x SL x NF -> (BS * SL) x NF
    next_features = pack_padded_sequence(
        next_features,
        lengths=seq_len,
        batch_first=True,
        enforce_sorted=False,
    )[0]

    # (BS * SL) x NF -> (BS * SL) x 1 x NF
    next_features = next_features.unsqueeze(-2)

    # (BS * SL) x NG x NF
    next_features_log_prob = Normal(mu, sigma).log_prob(next_features)

    # (BS * SL) x NG x NF -> (BS * SL) x NG
    next_features_log_prob_sum = next_features_log_prob.sum(dim=-1)

    loss = -torch.logsumexp(log_pi + next_features_log_prob_sum, dim=-1).mean()

    return loss
