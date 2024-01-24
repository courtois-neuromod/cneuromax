"""Typing utilities."""

from jaxtyping import Num
from torch import Tensor

Batch_type = (
    Num[Tensor, " ..."]
    | tuple[Num[Tensor, " ..."], ...]
    | list[Num[Tensor, " ..."]]
    | dict[str, Num[Tensor, " ..."]]
)
