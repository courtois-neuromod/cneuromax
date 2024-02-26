"""Typing utilities."""

from jaxtyping import Num
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding

Batch_type = (
    Num[Tensor, " ..."]
    | tuple[Num[Tensor, " ..."], ...]
    | list[Num[Tensor, " ..."]]
    | dict[str, Num[Tensor, " ..."]]
    | BatchEncoding
)
