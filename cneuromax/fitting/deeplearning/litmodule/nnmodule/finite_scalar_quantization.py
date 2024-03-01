import torch
import torch.nn.functional as f
from jaxtyping import Float
from torch import Tensor


class FiniteScalarQuantization(nn.Module):

    def __init__(self: "FiniteScalarQuantization", levels: list[int]) -> None:
        super().__init__()
        self.levels = torch.tensor(levels)
        if not (self.levels % 2 == 1).all():  # type: ignore[attr-defined]
            error_msg = "All levels must be odd."
            raise ValueError(error_msg)

    def bound(
        self: "FiniteScalarQuantization",
        z: Float[Tensor, " ..."],
    ) -> Float[Tensor, " ..."]:
        levels_max_values = (self.levels - 1) / 2
        out: Tensor = f.tanh(z) * levels_max_values
        return out

    def quantize(self, z: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        bounded_z = self.bound(z)
        rounded_bound_z = bounded_z.round()
        out = bounded_z + (rounded_bound_z - bounded_z).detach()
        levels_halved =
