""":mod:`~cneuromax.runner.BaseTaskRunner` utilities."""
import os
import sys
from importlib import import_module
from types import ModuleType
from typing import Any


def get_project_and_task_names() -> tuple[str, str]:
    """Retrieves ``project`` and ``task`` from script arguments.

    Raises:
        RuntimeError: If ``project`` or ``task`` arguments are\
            missing.

    Returns:
        The ``project`` and ``task`` names.
    """
    has_project_arg, has_task_arg = False, False
    for arg in sys.argv:
        if arg.startswith("project="):
            has_project_arg = True
            project_name = arg.split("=", maxsplit=1)[-1]
        if arg.startswith("task="):
            has_task_arg = True
            task_name = arg.split("=", maxsplit=1)[-1]
    if not has_project_arg:
        error_msg = (
            "Invalid script arguments. You must specify the "
            "``project`` argument in the form ``project=foo``."
        )
        raise RuntimeError(error_msg)
    if not has_task_arg:
        error_msg = (
            "Invalid script arguments. You must specify the "
            "``task`` argument in the form ``task=bar``."
        )
        raise RuntimeError(error_msg)
    return project_name, task_name


def get_absolute_project_path() -> str:
    """.

    Returns:
        The absolute path to the ``project`` module.
    """
    project_name, _ = get_project_and_task_names()
    return f"{os.environ['CNEUROMAX_PATH']}/cneuromax/projects/{project_name}/"


def get_project_module() -> ModuleType:
    """Retrieves the ``project`` module.

    Raises:
        RuntimeError: If the ``project`` argument is invalid or\
            the ``project`` module does not exist.

    Returns:
        The ``project`` module.
    """
    project_name, _ = get_project_and_task_names()
    try:
        project_module = import_module(f"cneuromax.projects.{project_name}")
    except ModuleNotFoundError as error:
        error_msg = (
            "Invalid project name. Make sure that "
            f"`cneuromax/projects/{project_name}/__init__.py` exists."
        )
        raise RuntimeError(error_msg) from error
    return project_module


def get_task_runner_class() -> Any:  # noqa: ANN401
    """.

    Raises:
        RuntimeError: If the ``project`` module does not define a\
            :mod:`~cneuromax.runner.BaseTaskRunner` class.

    Returns:
        The :mod:`~cneuromax.runner.BaseTaskRunner` class.
    """
    project_module = get_project_module()
    try:
        task_runner = project_module.TaskRunner
    except AttributeError as error:
        error_msg = (
            "Invalid project module. The ``project`` module must "
            "define a ``TaskRunner`` class."
        )
        raise RuntimeError(error_msg) from error
    return task_runner
