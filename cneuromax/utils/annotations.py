"""Type annotations validated through Beartype."""

from beartype.vale import Is
from beartype.vale._core._valecore import BeartypeValidator


def not_empty() -> BeartypeValidator:
    """Makes sure the string is not empty.

    Returns:
        .
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
        .
    """

    def _equal(x: object, element: object) -> bool:
        if x == element:
            return True
        return False

    return Is[lambda x: _equal(x, element)]


def one_of(*elements: object) -> BeartypeValidator:
    """Makes sure the value is one of the input arguments.

    Used to replace Typing ``Literal`` which is not supported by Hydra.

    Args:
        elements: The objects to compare against.

    Returns:
        .
    """

    def _one_of(x: object, elements: tuple[object, ...]) -> bool:
        if x in elements:
            return True
        return False

    return Is[lambda x: _one_of(x, elements)]


def has_keys(keys: list[str]) -> BeartypeValidator:
    """Makes sure the dictionary has the given keys.

    Args:
        keys: The keys to check for.

    Returns:
        .
    """

    def _has_keys(x: object, keys: list[str]) -> bool:
        if isinstance(x, dict):
            return all(key in x for key in keys)
        return False

    return Is[lambda x: _has_keys(x, keys)]


def ge(val: float) -> BeartypeValidator:
    """Validates greater than or equal to input argument.

    Args:
        val: The value to compare against.

    Returns:
        .
    """

    def _ge(x: object, val: float) -> bool:
        if isinstance(x, float) and x >= val:
            return True
        return False

    return Is[lambda x: _ge(x, val)]


def gt(val: float) -> BeartypeValidator:
    """Validates greater than input argument.

    Args:
        val: The value to compare against.

    Returns:
        .
    """

    def _gt(x: object, val: float) -> bool:
        if isinstance(x, float) and x > val:
            return True
        return False

    return Is[lambda x: _gt(x, val)]


def le(val: float) -> BeartypeValidator:
    """Validate less than or equal to input argument.

    Args:
        val: The value to compare against.

    Returns:
        .
    """

    def _le(x: object, val: float) -> bool:
        if isinstance(x, float) and x <= val:
            return True
        return False

    return Is[lambda x: _le(x, val)]


def lt(val: float) -> BeartypeValidator:
    """Validate less than input argument.

    Args:
        val: The value to compare against.

    Returns:
        .
    """

    def _lt(x: object, val: float) -> bool:
        if isinstance(x, float) and x < val:
            return True
        return False

    return Is[lambda x: _lt(x, val)]
