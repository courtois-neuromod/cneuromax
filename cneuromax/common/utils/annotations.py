"""."""

from typing import Annotated, TypeAlias

from beartype.vale import Is

non_empty_str: TypeAlias = Annotated[str, Is[lambda x: len(x) > 0]]

str_is_per_device_batch_size: TypeAlias = Annotated[
    str,
    Is[lambda x: x in ("per_device_batch_size")],
]
"""String ``is "per_device_batch_size"``."""


str_is_fit_or_test: TypeAlias = Annotated[
    str,
    Is[lambda x: x in ("fit", "test")],
]
"""String ``in ("fit", "test")``."""

str_is_cpu_or_gpu: TypeAlias = Annotated[
    str,
    Is[lambda x: x in ("cpu", "gpu")],
]
"""String ``in ("cpu", "gpu")``."""

int_is_gt0: TypeAlias = Annotated[int, Is[lambda x: x > 0]]
"""Integer ``> 0``."""

int_is_ge0: TypeAlias = Annotated[int, Is[lambda x: x > 0]]
"""Integer ``>= 0``."""

float_is_ge0_le1: TypeAlias = Annotated[float, Is[lambda x: 0 <= x <= 1]]
"""Float ``>= 0`` and ``<= 1``."""
