"""Miscellaneous utilities."""
from collections.abc import Callable
from typing import Any


def get_path(clb: Callable[..., Any]) -> str:
    """Returns the path to the input callable.

    Args:
        clb: The callable to retrieve the path for.

    Returns:
        The full module path to :paramref:`clb`.
    """
    return f"{clb.__module__}.{clb.__name__}"
