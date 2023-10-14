"""Hydra-related utilities."""

from collections.abc import Callable
from typing import Any

from hydra_zen import make_custom_builds_fn


def get_path(clb: Callable[..., Any]) -> str:
    """Returns path to input class/function.

    Used to fill in ``_target_`` in Hydra configuration.

    Args:
        clb: Class or function.

    Returns:
        Path to class or function.
    """
    return f"{clb.__module__}.{clb.__name__}"


fs_builds = make_custom_builds_fn(populate_full_signature=True)
pfs_builds = make_custom_builds_fn(
    populate_full_signature=True,
    zen_partial=True,
)
