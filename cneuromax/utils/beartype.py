"""Type annotations validator using :mod:`beartype`."""

from beartype.vale import Is
from beartype.vale._core._valecore import BeartypeValidator


def not_empty() -> BeartypeValidator:
    """Makes sure the string is not empty.

    Returns:
        A
            :mod:`beartype` object that raises an exception if the
            annotated value does not satisfy the condition.
    """

    def _not_empty(x: object) -> bool:
        return isinstance(x, str) and len(x) > 0

    return Is[lambda x: _not_empty(x)]


def equal(element: object) -> BeartypeValidator:
    """Verifies that the annotated value is equal to the input argument.

    Args:
        element: The object to compare the annotated value against.

    Returns:
        See return description of :func:`not_empty`.
    """

    def _equal(x: object, element: object) -> bool:
        return x == element

    return Is[lambda x: _equal(x, element)]


def one_of(*elements: object) -> BeartypeValidator:
    """Verifies that the annotated value is one of the input arguments.

    Used to replace :class:`typing.Literal` which is not supported by
    :mod:`omegaconf`-based configs.

    Args:
        elements: The objects to compare the annotated value against.

    Returns:
        See return description of :func:`not_empty`.
    """

    def _one_of(x: object, elements: tuple[object, ...]) -> bool:
        return x in elements

    return Is[lambda x: _one_of(x, elements)]


def ge(val: float) -> BeartypeValidator:
    """Verifies that the annotated value is ``> or =`` :paramref:`val`.

    Args:
        val: The value to compare the annotated value against.

    Returns:
        See return description of :func:`not_empty`.
    """

    def _ge(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x >= val

    return Is[lambda x: _ge(x, val)]


def gt(val: float) -> BeartypeValidator:
    """Verifies that the annotated value is ``>`` :paramref:`val`.

    Args:
        val: See :paramref:`~ge.val`.

    Returns:
        See return description of :func:`not_empty`.
    """

    def _gt(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x > val

    return Is[lambda x: _gt(x, val)]


def le(val: float) -> BeartypeValidator:
    """Verifies that the annotated value is ``< or =`` :paramref:`val`.

    Args:
        val: See :paramref:`~ge.val`.

    Returns:
        See return description of :func:`not_empty`.
    """

    def _le(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x <= val

    return Is[lambda x: _le(x, val)]


def lt(val: float) -> BeartypeValidator:
    """Verifies that the annotated value is ``<`` :paramref:`val`.

    Args:
        val: See :paramref:`~ge.val`.

    Returns:
        See return description of :func:`not_empty`.
    """

    def _lt(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x < val

    return Is[lambda x: _lt(x, val)]
