"""Tests for :mod:`cneuromax.utils.misc`."""
from unittest.mock import patch

from cneuromax.utils.misc import get_path, get_project_path, get_script_path


def test_get_path() -> None:
    """Tests for :func:`cneuromax.utils.misc.get_path`."""
    assert get_path(get_path) == "cneuromax.utils.misc.get_path"


def test_get_script_path() -> None:
    """Tests for :func:`cneuromax.utils.misc.get_script_path`."""
    sys_argv = ["-m", "foo"]
    with patch("sys.argv", sys_argv):
        assert get_script_path() == "foo.__main__"
    sys_argv = ["foo.py"]
    with patch("sys.argv", sys_argv):
        assert get_script_path() == "foo"
    sys_argv = ["foo/bar.py"]
    with patch("sys.argv", sys_argv):
        assert get_script_path() == "foo.bar"


def test_get_project_path() -> None:
    """Tests for :func:`cneuromax.utils.misc.get_project_path`."""
    sys_argv = ["-m", "foo"]
    with patch("sys.argv", sys_argv):
        assert get_project_path() == "foo"
    sys_argv = ["test/foo.py"]
    with patch("sys.argv", sys_argv):
        assert get_project_path() == "test"
    sys_argv = ["foo/bar/baz/__main__.py", "task=qux"]
    with patch("sys.argv", sys_argv):
        assert get_project_path() == "foo.bar.baz"
