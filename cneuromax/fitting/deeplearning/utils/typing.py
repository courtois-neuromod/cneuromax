"""Typing utilities."""

from jaxtyping import Num
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding

Batched_data_type = (
    Num[Tensor, " BS *_"]
    | tuple[Num[Tensor, " BS *_"], ...]
    | list[Num[Tensor, " BS *_"]]
    | dict[str, Num[Tensor, " BS *_"]]
    | BatchEncoding
)
"""Type hint for batched data.

BS: batch size.
"""
