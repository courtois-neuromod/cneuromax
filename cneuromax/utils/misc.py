"""Miscellaneous utilities."""

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import requests  # type: ignore[import-untyped]
import torch


def get_path(clb: Callable[..., Any]) -> str:
    """Returns the path to the input callable.

    Args:
        clb: The callable to retrieve the path for.

    Returns:
        The full module path to :paramref:`clb`.
    """
    return f"{clb.__module__}.{clb.__name__}"


def seed_all(seed: int | np.uint32) -> None:
    """Sets the random seed for all relevant libraries.

    Args:
        seed: The random seed.
    """
    random.seed(a=int(seed))
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=int(seed))
    torch.cuda.manual_seed_all(seed=int(seed))


def can_connect_to_internet() -> bool:
    """Checks whether an internet connection is available."""
    try:
        response = requests.get(url="https://www.google.com", timeout=5)
        response.raise_for_status()
    except Exception:  # noqa: BLE001
        return False
    return True
