""":class:`.CPUStaticRNN` & its config."""

from dataclasses import dataclass

import torch
from jaxtyping import Float32
from torch import Tensor, nn


@dataclass
class CPUStaticRNNFCConfig:
    """Config values for :class:`CPUStaticRNNFC`.

    Args:
        input_size: Size of the input tensor.
        hidden_size: Size of the RNN hidden state.
        output_size: Size of the output tensor.
    """

    input_size: int
    hidden_size: int
    output_size: int


class CPUStaticRNNFC(nn.Module):
    """CPU-running static architecture RNN w/ a final FC layer.

    Args:
        config
    """

    def __init__(self: "CPUStaticRNNFC", config: CPUStaticRNNFCConfig) -> None:
        super().__init__()
        self.rnn = nn.RNNCell(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
        )
        self.fc = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.output_size,
        )
        self.h: Float32[Tensor, " hidden_size"] = torch.zeros(
            size=(config.hidden_size,),
        )
        for param in self.parameters():
            param.requires_grad = False
            param.data = torch.zeros_like(param.data)

    def reset(self: "CPUStaticRNNFC") -> None:
        """Resets the hidden state of the RNN."""
        self.h *= torch.zeros_like(self.h)

    def forward(
        self: "CPUStaticRNNFC",
        x: Float32[Tensor, " input_size"],
    ) -> Tensor:
        """Forward pass."""
        x: Float32[Tensor, " hidden_size"] = self.rnn(input=x, hx=self.h)
        self.h = x
        x: Float32[Tensor, " output_size"] = self.fc(input=x)
        return x
