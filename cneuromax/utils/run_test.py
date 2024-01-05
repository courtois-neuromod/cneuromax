"""Tests for :mod:`cneuromax.utils.run`."""
from unittest.mock import patch

from cneuromax.utils.run import (
    parse_task_argument,
    retrieve_module_functions,
)


def test_parse_task_argument() -> None:
    """Test :func:`~.run.parse_task_argument`."""
    # Expected regular behavior:
    test_args = [
        "-m",
        "cneuromax",
        "task=fitting/deeplearning/classify_mnist/mlp",
    ]
    with patch("sys.argv", test_args):
        service_name, project_name, task_name = parse_task_argument()
        assert service_name == "fitting/deeplearning"
        assert project_name == "classify_mnist"
        assert task_name == "mlp"
    # Expected more complex behavior:
    test_args = [
        "-m",
        "cneuromax",
        "seed=0",
        "task=service_name_0/service_name_1/service_name_2/project_name/task_name",
    ]
    with patch("sys.argv", test_args):
        service_name, project_name, task_name = parse_task_argument()
        assert service_name == "service_name_0/service_name_1/service_name_2"
        assert project_name == "project_name"
        assert task_name == "task_name"
    # Missing `task` argument:
    test_args = [
        "-m",
        "cneuromax",
        "seed=0",
    ]
    with patch("sys.argv", test_args):
        try:
            parse_task_argument()
        except ValueError:
            pass
        else:
            raise AssertionError
    # Missing `=` in `task` argument:
    test_args = [
        "-m",
        "cneuromax",
        "seed=0",
        "taskservice_name_0/service_name_1/service_name_2/project_name",
    ]
    with patch("sys.argv", test_args):
        try:
            parse_task_argument()
        except ValueError:
            pass
        else:
            raise AssertionError
    # `task` argument not complete #1
    test_args = [
        "-m",
        "cneuromax",
        "seed=0",
        "task=service_name",
    ]
    with patch("sys.argv", test_args):
        try:
            parse_task_argument()
        except ValueError:
            pass
        else:
            raise AssertionError
    # `task` argument not complete #2
    test_args = [
        "-m",
        "cneuromax",
        "seed=0",
        "task=service_name/project_name",
    ]
    with patch("sys.argv", test_args):
        try:
            parse_task_argument()
        except ValueError:
            pass
        else:
            raise AssertionError


def test_retrieve_module_functions() -> None:
    """Test :func:`.retrieve_module_functions`."""
    # Expected regular behavior:
    retrieve_module_functions("fitting/deeplearning")
    retrieve_module_functions("fitting/neuroevolution")
    retrieve_module_functions("fitting/hybrid")
    # Expected error:
    try:
        retrieve_module_functions("fitting/randomsearch")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError
