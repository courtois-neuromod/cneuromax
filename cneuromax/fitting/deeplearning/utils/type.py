"""Typing utilities."""

from jaxtyping import Num
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding

Batched_data_type = (
    Num[Tensor, " batch_size ..."]
    | tuple[Num[Tensor, " batch_size ..."], ...]
    | list[Num[Tensor, " batch_size ..."]]
    | dict[str, Num[Tensor, " batch_size ..."]]
    | BatchEncoding
)
