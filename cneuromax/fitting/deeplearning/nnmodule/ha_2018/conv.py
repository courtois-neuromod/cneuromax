import torch
import torch.nn as nn
from jaxtyping import Float


class Ha2018ConvNet(nn.Module):
    """ConvNet as seen in the "World Models" paper (Ha & Schmidhuber, 2018)
    https://arxiv.org/abs/1803.10122
    """

    def __init__(self, output_size: int) -> None:
        """Constructor.

        Args:
            output_size: Size of the batched output vectors.
        """
        assert isinstance(output_size, int) and output_size >= 1

        self.model = nn.Sequential(
            # BS x 3 x 64 x 64 -> BS x 32 x 31 x 31
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            # BS x 32 x 31 x 31 -> BS x 64 x 14 x 14
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            # BS x 64 x 14 x 14 -> BS x 128 x 6 x 6
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            # BS x 128 x 6 x 6 -> BS x 256 x 2 x 2
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
            # BS x 256 x 2 x 2 -> BS x 1024
            nn.Flatten(),
            # BS x 1024 -> BS x output_size
            nn.Linear(1024, output_size),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size 3 64 64"]
    ) -> Float[torch.Tensor, "batch_size output_size"]:
        """Forward method.

        Args:
            x: The batched input images.
        Returns:
            The batched output vectors.
        """
        return self.model(x)


class Ha2018TConvNet(nn.Module):
    """Transposed ConvNet from the "World Models" paper (Ha & Schmidhuber, 2018)

    https://arxiv.org/abs/1803.10122
    """

    def __init__(self, input_size: int) -> None:
        """Constructor.

        Args:
            input_size: Size of the batched input vectors.
        """
        assert isinstance(input_size, int) and input_size >= 1

        self.model = nn.Sequential(
            # BS x input_size -> BS x 1024
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            # BS x 1024 -> BS x 1024 x 1 x 1
            nn.Unflatten(),
            # BS x 1024 x 1 x 1 -> BS x 128 x 5 x 5
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(),
            # BS x 128 x 5 x 5 -> BS x 64 x 13 x 13
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            # BS x 64 x 13 x 13 -> BS x 32 x 30 x 30
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            # BS x 32 x 30 x 30 -> BS x 3 x 64 x 64
            nn.ConvTranspose2d(32, 3, 6, stride=2),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size input_size"]
    ) -> Float[torch.Tensor, "batch_size 3 64 64"]:
        """Forward method.

        Args:
            x: The batched input vectors.
        Returns:
            The batched image reconstructions.
        """
        return self.model(x)
