"""."""

from typing import Annotated

from beartype.vale import Is

float_gt0_lt1 = Annotated[float, Is[lambda x: 0 < x < 1]]
str_fit_test = Annotated[str, Is[lambda x: x in ("fit", "test")]]
