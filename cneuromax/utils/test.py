"""Custom test utilities."""

import inspect
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def is_decorated_with(method: Callable[..., T], string: str) -> bool:
    """Check if a method is decorated with ``@{string}``.

    Args:
        method: The callable to check.
        string: The string to check for.

    Returns:
        True if the method is decorated with ``@{string}``, False
            otherwise.
    """
    source_lines, _ = inspect.getsourcelines(method)

    for line in source_lines:
        if "def" in line:
            return False
        if f"@{string}" in line:
            return True

    return False
