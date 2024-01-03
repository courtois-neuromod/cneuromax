"""Miscellaneous utilities for :mod:`cneuromax`."""

import inspect
import sys
from collections.abc import Callable
from importlib import import_module
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


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


def retrieve_module_functions(
    service_name: str,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Returns ``run`` & ``store_configs`` funcs from ``service_name``.

    In order for the module specified with :paramref:`service_name` to be
    valid, it must define both a :func:`hydra.main` decorated
    :func:`run` function that details its execution and a
    :func:`store_configs` function that stores its structured
    :mod:`hydra-core` configurations. More details on how to define
    these functions can be found in the root :mod:`cneuromax` API
    documentation.

    Args:
        service_name: The string parsed from the command-line arguments\
            in :func:`parse_task_argument` that specifies the project's\
            service name.

    Returns:
        * The :func:`store_configs` function of :paramref:`service_name`\
            turned into a Python module.
        * The :func:`run` function of :paramref:`service_name` turned\
            into a Python module.

    Raises:
        ModuleNotFoundError: If :paramref:`service_name` is not a valid
            module.
        NotImplementedError: If :paramref:`service_name` does not\
            define both a :func:`run` and a :func:`store_configs`\
            function.
    """
    try:
        service_name_module = import_module(
            f"cneuromax.task.{service_name}",
        )
    except ModuleNotFoundError as e:
        error_msg = f"Service name '{service_name}' does not exist. "
        raise ModuleNotFoundError(error_msg) from e
    if not hasattr(service_name_module, "run") or not callable(
        service_name_module.run,
    ):
        error_msg = (
            f"File '{service_name}'.__init__.py does not define a "
            "callable `run` function. Check-out the root `cneuromax` API "
            "documentation for more information on how to properly define "
            "this function."
        )
        raise NotImplementedError(error_msg)

    def dummy_run_fn(config: DictConfig) -> None:  # noqa: ARG001
        return

    if inspect.signature(obj=service_name_module.run) != inspect.signature(
        obj=dummy_run_fn,
    ):
        error_msg = (
            f"File '{service_name}.__init__.py' `run` function signature "
            "does not match the expected signature. Check-out the root "
            "`cneuromax` API documentation for more information on how to "
            "properly define this function."
        )
        raise NotImplementedError(error_msg)
    if not hasattr(service_name_module, "store_configs") or not callable(
        service_name_module.store_configs,
    ):
        error_msg = (
            f"File '{service_name}.__init__.py' does not define a "
            "callable `store_configs` function. Check-out the root "
            "`cneuromax` API documentation for more information on how to "
            "properly define this function."
        )
        raise NotImplementedError(error_msg)

    def dummy_store_configs_fn(cs: ConfigStore) -> None:  # noqa: ARG001
        return

    if inspect.signature(
        obj=service_name_module.store_configs,
    ) != inspect.signature(obj=dummy_store_configs_fn):
        error_msg = (
            f"File '{service_name}.__init__.py' `store_configs` function "
            "signature does not match the expected signature. Check-out the "
            "root `cneuromax` API documentation for more information on how "
            "to properly define this function."
        )
        raise NotImplementedError(error_msg)
    return service_name_module.store_configs, service_name_module.run
