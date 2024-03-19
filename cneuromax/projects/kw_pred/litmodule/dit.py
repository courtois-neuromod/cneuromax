"""."""

from typing import Any

from torch import Tensor, nn

from cneuromax.projects.kw_pred.DiT.models import DiT


class CustomDiT(DiT):
    """Custom DiT model."""

    def __init__(
        self: "CustomDiT",
        input_size: int = 32,
        hidden_size: int = 1152,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the model."""
        super().__init__(input_size, hidden_size, *args, **kwargs)  # type: ignore[no-untyped-call]

        class TransposeMin1Min2(nn.Module):
            def forward(self: "TransposeMin1Min2", x: Tensor) -> Tensor:
                return x.transpose(-1, -2)

        self.x_embedder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=hidden_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            TransposeMin1Min2(),
        )  # BS x in_channels x seq_len -> BS x num_patches x hidden_size
