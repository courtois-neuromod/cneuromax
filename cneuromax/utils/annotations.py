"""Custom annotations for type checking."""

from typing import Annotated

from beartype.vale import Is

int_gt_0 = Annotated[int, Is[lambda x: x > 0]]
int_ge_0 = Annotated[int, Is[lambda x: x >= 0]]

float_gt_0_lt_1 = Annotated[float, Is[lambda x: 0 < x < 1]]

str_cpu_or_gpu = Annotated[str, Is[lambda x: x in ("cpu", "gpu")]]
str_fit_or_test = Annotated[str, Is[lambda x: x in ("fit", "test")]]
str_per_device_batch_size = Annotated[
    str,
    Is[lambda x: x in ("per_device_batch_size")],
]
