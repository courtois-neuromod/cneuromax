"""Miscellaneous utilities for :mod:`cneuromax`."""
import sys


def parse_task_argument() -> tuple[str, str, str]:
    """Parses the ``task`` argument from the command-line arguments.

    An example valid ``task`` argument is
    `task=fitting/deeplearning/classify_mnist/mlp` where
    `fitting/deeplearning` is the service name, ``classify_mnist`` is
    the project name and ``mlp`` is the task name.

    Returns:
        * The service name.
        * The project name.
        * The task name.
    """
    error_msg = (
        "The `task` argument must be of the form "
        "`task=fitting/deeplearning/classify_mnist/mlp` "
        "where `fitting/deeplearning` is the service name, `classify_mnist` "
        "is the project name and `mlp` is the task name."
    )
    for arg in sys.argv:
        # e.g. `task=fitting/deeplearning/classify_mnist/mlp`
        if arg.startswith("task="):
            # e.g. `fitting/deeplearning/classify_mnist/mlp``
            if len(arg.split(sep="=", maxsplit=1)) != 2:  # noqa: PLR2004
                raise ValueError(error_msg)
            full_task_name = arg.split(sep="=", maxsplit=1)[1]
            if (
                len(full_task_name.rsplit(sep="/", maxsplit=1))
                != 2  # noqa: PLR2004
            ):
                raise ValueError(error_msg)
            full_project_name, task_name = full_task_name.rsplit(
                sep="/",
                maxsplit=1,
            )
            if (
                len(full_project_name.rsplit(sep="/", maxsplit=1))
                != 2  # noqa: PLR2004
            ):
                raise ValueError(error_msg)
            service_name, project_name = full_project_name.rsplit(
                sep="/",
                maxsplit=1,
            )
            return service_name, project_name, task_name
    raise ValueError(error_msg)
