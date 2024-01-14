"""Miscellaneous utilities."""
import sys
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


def get_script_path() -> str:
    """Returns the path to the executed script.

    Returns:
        The full module path to the executed script in the form
        ``foo.bar.baz``.
    """
    # Loop while the first arguments are command flags.
    i = 0
    while i < len(sys.argv) and sys.argv[i].startswith("-"):
        i += 1
    script_path = sys.argv[i]
    if script_path.endswith(".py"):
        script_path = script_path.replace(".py", "")
        script_path = script_path.replace("/", ".")
    else:  # Calling a `__main__.py` file.
        script_path = script_path + ".__main__"
    return script_path


def get_project_path() -> str:
    """Returns the path to the current ``project``.

    Returns:
        The path to the current ``project`` in the form ``foo.bar.baz``.
    """
    return get_script_path().rsplit(".", maxsplit=1)[0]
