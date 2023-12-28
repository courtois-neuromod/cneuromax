"""Type annotations validator through :mod:`beartype`."""

from beartype.vale import Is
from beartype.vale._core._valecore import BeartypeValidator


def not_empty() -> BeartypeValidator:
    """Makes sure the string is not empty.

    Returns:
        The corresponding validator.
    """

    def _not_empty(x: object) -> bool:
        if isinstance(x, str) and len(x) > 0:
            return True
        return False

    return Is[lambda x: _not_empty(x)]


def equal(element: object) -> BeartypeValidator:
    """Makes sure the value is equal to the input argument.

    Args:
        element: The object to compare against.

    Returns:
        The corresponding validator.
    """

    def _equal(x: object, element: object) -> bool:
        if x == element:
            return True
        return False

    return Is[lambda x: _equal(x, element)]


def one_of(*elements: object) -> BeartypeValidator:
    """Makes sure the value is one of the input arguments.

    Used to replace :class:`typing.Literal` which is not supported by
    :mod:`omegaconf`-based configs.

    Args:
        elements: The objects to compare against.

    Returns:
        The corresponding validator.
    """

    def _one_of(x: object, elements: tuple[object, ...]) -> bool:
        if x in elements:
            return True
        return False

    return Is[lambda x: _one_of(x, elements)]


def ge(val: float) -> BeartypeValidator:
    """Makes sure the input is greater of equal than ``val``.

    Args:
        val: The value to compare against.

    Returns:
        The corresponding validator.
    """

    def _ge(x: object, val: float) -> bool:
        if isinstance(x, float) and x >= val:
            return True
        return False

    return Is[lambda x: _ge(x, val)]


def gt(val: float) -> BeartypeValidator:
    """Makes sure the input is greater than ``val``.

    Args:
        val: The value to compare against.

    Returns:
        The corresponding validator.
    """

    def _gt(x: object, val: float) -> bool:
        if isinstance(x, float) and x > val:
            return True
        return False

    return Is[lambda x: _gt(x, val)]


def le(val: float) -> BeartypeValidator:
    """Makes sure the input is less or equal than ``val``.

    Args:
        val: The value to compare against.

    Returns:
        The corresponding validator.
    """

    def _le(x: object, val: float) -> bool:
        if isinstance(x, float) and x <= val:
            return True
        return False

    return Is[lambda x: _le(x, val)]


def lt(val: float) -> BeartypeValidator:
    """Makes sure the input is less than ``val``.

    Args:
        val: The value to compare against.

    Returns:
        The corresponding validator.
    """

    def _lt(x: object, val: float) -> bool:
        if isinstance(x, float) and x < val:
            return True
        return False

    return Is[lambda x: _lt(x, val)]
