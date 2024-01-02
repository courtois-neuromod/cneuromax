""":mod:`cneuromax` codebase.

.. note::

    In the following definitions ``...`` corresponds to the type of\
    task you are performing. Examples include ``serving``,\
    ``fitting/deeplearning``, etc. See Section II for an example.

I. Terminology specific to the :mod:`cneuromax` codebase used\
throughout the documentation:

- ``project``: A single/multi-person endeavor with a\
    corresponding ``cneuromax/task/.../PROJECT_NAME/`` sub-directory.\
    This sub-directory needs to contain at the very least a\
    :mod:`hydra-core` ``TASK_NAME.yaml`` file to specify a runnable\
    ``task`` (see below). In the case that the project defines some\
    Python modules, the sub-directory's ``__init__.py`` should\
    optionally (but strongly recommended) define a ``store_configs``\
    function that registers the project's :mod:`hydra-core`\
    configurations to simplify your ``.yaml`` ``task`` configuration\
    files.
- ``task``: A work unit specified by a :mod:`hydra-core` configuration\
    ``.yaml`` file located in a ``project`` subdirectory. A ``task``\
    with a corresponding\
    ``cneuromax/task/.../PROJECT_NAME/TASK_NAME.yaml``\
    file can be executed by running\
    ``python -m cneuromax task=.../PROJECT_NAME/TASK_NAME``. Not to be\
    confused with SLURM's ``task`` terminology, which refers to the\
    single or multi-process execution of a specific script.

II. MNIST classification example:

`Source files <https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/task/fitting/deeplearning/classify_mnist>`_

Usage:

.. note::

    Do not forget to prepend the Docker/Apptainer command.

.. note::

    Do not include the ``.yaml`` extension at the end of the\
    ``task`` argument.

``python -m cneuromax task=fitting/deeplearning/classify_mnist/mlp``

III. ``.yaml`` & ``__main__.py`` files:

``base_config.yaml``:

.. highlight:: yaml
.. code-block:: yaml

    hydra:
        callbacks:
            log_job_return:
                _target_: >
                    hydra.experimental.callbacks.LogJobReturnCallback
        job:
            config:
                override_dirname:
                    kv_sep: '.'
                    item_sep: '~'
                    exclude_keys:
                        - task
        mode: MULTIRUN
        searchpath:
            - file://${oc.env:CNEUROMAX_PATH}/cneuromax/
        sweep:
            dir: >
            ${oc.env:CNEUROMAX_PATH}/cneuromax/data/task_run/${hydra.runtime.choices.task}/
            subdir: ${hydra.job.override_dirname}/
    defaults:
        - _self_
        - task: null
"""

import logging
import sys
import warnings
from importlib import import_module

from beartype import BeartypeConf
from beartype.claw import beartype_this_package
from hydra.core.config_store import ConfigStore

beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))
warnings.filterwarnings("ignore", module="beartype")
warnings.filterwarnings("ignore", module="lightning")


def store_project_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` project configurations.

    Parses the project path from the script arguments, import
    its ``store_configs`` function and calls it.

    Args:
        cs: A singleton instance that manages the :mod:`hydra-core`\
            configuration store.
    """
    for arg in sys.argv:
        # e.g. `task=fitting/deeplearning/classify_mnist/mlp`
        if "task=" in arg[:5] and len(arg) > 5:  # noqa: PLR2004
            try:
                # e.g. `fitting/deeplearning/classify_mnist/mlp``
                full_task_name = arg.split(sep="=", maxsplit=1)[1]
                # e.g. `fitting/deeplearning/classify_mnist``
                full_project_name = full_task_name.rsplit(sep="/", maxsplit=1)[
                    0
                ]
                project_module = import_module(
                    f"cneuromax.task.{full_project_name}",
                )
            except ModuleNotFoundError:
                logging.warning(
                    f"The `__init__.py' file of project '{full_project_name}'"
                    "cannot be found. Make sure it exists at "
                    f"`cneuromax/task/{full_project_name}/__init__.py` and is "
                    "spelled correctly."
                    "Check-out "
                    "`cneuromax/task/fitting/deeplearning/classify_mnist/__init__.py`"
                    "for an example.",
                )
                return
            try:
                project_module.store_configs(cs)
            except AttributeError:
                logging.exception(
                    "The project module `store_configs` function cannot be "
                    "found. Make sure it exists in your "
                    f"`cneuromax/task/{full_project_name}/__init__.py` file. "
                    "Check-out "
                    "`cneuromax/task/fitting/deeplearning/classify_mnist/__init__.py`"
                    "for an example.",
                )
                return
    module_not_found_error_2 = (
        "The task must be specified in the script arguments. "
        "Check-out "
        "`cneuromax/task/fitting/deeplearning/classify_mnist/__init__.py`"
        "for an example.",
    )
    raise ModuleNotFoundError(module_not_found_error_2)
