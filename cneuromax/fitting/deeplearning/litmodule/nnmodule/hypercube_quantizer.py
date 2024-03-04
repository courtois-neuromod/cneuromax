from typing import Annotated as An

import torch
import torch.nn.functional as f
from einops import repeat
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from cneuromax.utils.beartype import ge


class HypercubeQuantizer(nn.Module):
    """``Finite Scalar Quantization`` with dimensions of fixed size 8.

    Inspired from: Finite Scalar Quantization: VQ-VAE Made Simple
    Paper link: https://arxiv.org/abs/2309.15505
    Authors: Mentzer et al. (2023)

    Benefits of having dimensions of fixed size 8:
    1) Makes the implementation even simpler
    2) Is over 5 which appears to be the minimum size to not decrease
    performance in the paper experiments (Figure 10).
    3) Simple binary transformation: 0 -> 000, 1 -> 001, ..., 7 -> 111

    Drawbacks:
    1) Cannot let the codebook size be fine-grained (increases in powers
    of 8).
    2) Assuming Mentzer et al. (2023) sweeped over dimension sizes,
    there could be benefits to choosing lower dimensions sizes (e.g. 5)
    and increasing the number of dimensions to reach the same number of
    codes.

    Args:
        num_levels: Number of levels for the quantizer.
            ``1 <=> size 2^3, 2 <=> size 2^6, 3 <=> size 2^9, ...``
    """

    def __init__(
        self: "HypercubeQuantizer",
        num_dims: An[int, ge(1)],
    ) -> None:
        super().__init__()
        self.num_dims = num_dims
        """
        num_levels = 4 -> 64, 8, 1
        self.multipliers = torch.tensor(
            data=[8**i for i in range(num_dims - 1, -1, -1)],
        )
        """

    def quantize(
        self: "HypercubeQuantizer",
        z: Float[Tensor, " batch_size ... num_dims"],
        *,
        normalize: bool = True,
    ) -> Float[Tensor, " batch_size ... num_dims"]:
        """Quantizes the input tensor.

        Args:
            z: Input tensor to be quantized.
            normalize: Whether to normalize the quantized tensor to
                the range [-1, 1] instead of [0, 7].

        Returns:
            The quantized tensor.
        """
        # ? -> [-1, 1]
        bounded_z: Float[Tensor, " BS ... ND"] = torch.tanh(z)
        # [-1, 1] -> [0, 7]
        scaled_z: Float[Tensor, " BS ... ND"] = bounded_z.add(1).mul(3.5)
        # [0, 7] -> 0, 1, 2, 3, 4, 5, 6, 7
        z_hat: Float[Tensor, " BS ... ND"] = scaled_z.round()
        z_hat_ste: Float[Tensor, " BS ... ND"] = (
            scaled_z + (z_hat - scaled_z).detach()
        )
        if normalize:
            # 0, 1, 2, 3, 4, 5, 6, 7
            # -> -1, -.6667, -.3333, 0, .3333, .6667, 1
            z_hat_ste = z_hat_ste.mul(2 / 7).sub(1)
        return z_hat_ste

    def quantize_to_binary(
        self: "HypercubeQuantizer",
        z: Float[Tensor, " batch_size ... num_dims"],
    ) -> Bool[Tensor, " batch_size ... num_dims_times_3"]:
        """Quantizes the input tensor to binary.

        Args:
            z: Input tensor to be quantized.

        Returns:
            The quantized tensor in binary.
        """
        # 0, 1, 2, 3, 4, 5, 6, 7
        z_hat_ste: Float[Tensor, " BS ... ND"] = self.quantize(
            z=z,
            normalize=False,
        )
        z_hat_ste_int: Int[Tensor, " BS ... ND 1"] = repeat(
            tensor=z_hat_ste.int(),
            pattern=" ... -> ... 1",
        )
        # 4, 2, 1
        mask = 2 ** torch.arange(start=2, end=-1, step=-1)
        # 2 -> 010, 6 -> 110
        binary_z_hat_ste: Bool[Tensor, " BS ...  ND 3"] = (
            z_hat_ste_int.bitwise_and(mask).ne(0)
        )
        return binary_z_hat_ste
