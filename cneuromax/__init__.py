""":mod:`cneuromax` package.

Terminology
==============

1. Quck definitions
~~~~~~~~~~~~~~~~~~~

**Task**: Some work unit specified by a :mod:`hydra-core`
config ``.yaml`` file that specifies its execution.

**Project**: A collection of ``tasks`` + cross-``task``
functionality (ex: a custom :mod:`lightning` ``datamodule``).

**Service**: A collection of ``projects`` +
cross-``project`` functionality (ex: base :mod:`lightning` sub-classes).

2. Service
~~~~~~~~~~

a. Service overview
-------------------

A ``service`` refers to the combination of:

- a package located at ``cneuromax/SERVICE_NAME/`` containing
  code common to multiple ``projects``.
- a package located at ``cneuromax/task/SERVICE_NAME/``
  containing ``projects``.

.. note::

    ``SERVICE_NAME`` can be a nested directory.

b. Example services
-------------------

Deep Net Training with ``fitting/deeplearning``:

- :mod:`cneuromax.fitting.deeplearning` (`source folder <https://github.com/\
    courtois-neuromod/cneuromax/tree/main/cneuromax/fitting/deeplearning>`_)
- :mod:`cneuromax.task.fitting.deeplearning` (`source folder <https://github.com/\
    courtois-neuromod/cneuromax/tree/main/cneuromax/task/fitting/deeplearning>`_)

Neural Net Evolution with ``fitting/neuroevolution``:

- :mod:`cneuromax.fitting.neuroevolution` (`source folder <https://github.com/\
    courtois-neuromod/cneuromax/tree/main/cneuromax/fitting/neuroevolution>`_)
- :mod:`cneuromax.task.fitting.neuroevolution` (`source folder <https://github.com/\
    courtois-neuromod/cneuromax/tree/main/cneuromax/task/fitting/neuroevolution>`_)

.. note::

    The above ``services`` are nested in a ``fitting``
    package, which is not a service but rather an
    "abstract" base for ``services``.

Model analysis with ``testing`` (in progress):

- :mod:`cneuromax.testing` (`source folder <https://github.com/\
    courtois-neuromod/cneuromax/tree/main/cneuromax/testing>`_)
- :mod:`cneuromax.task.testing` (`source folder <https://github.com/\
    courtois-neuromod/cneuromax/tree/main/cneuromax/task/testing>`_)

c. Creating new services
------------------------

.. note::

    Skip to `3. Project`_ if you do not intend to create new
    ``services`` (most contributors will not).

i. Create a ``cneuromax/task/SERVICE_NAME/`` directory
......................................................

Example: ``cneuromax/task/fitting/deeplearning`` (`source folder
<https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/task/fitting/deeplearning>`_)

ii. Create a ``__init__.py`` file in ``cneuromax/task/SERVICE_NAME/``
...............................................................

Example: ``cneuromax/task/fitting/deeplearning/__init__.py``
(`source file
<https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/task/fitting/deeplearning/__init__.py>`_)

.. warning::

    In order to be documented, all Python packages must contain a
    ``__init__.py`` file with at least one statement (``pass`` will do).

iii. Create a ``cneuromax/SERVICE_NAME/`` directory
...................................................

Example: ``cneuromax/fitting/deeplearning`` (`source folder
<https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/fitting/deeplearning>`_)

iv. Create a ``config.yaml`` file in ``cneuromax/SERVICE_NAME/``
...............................................................

Example: ``cneuromax/fitting/deeplearning/config.yaml`` (`source file
<https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/fitting/deeplearning/config.yaml>`_)

Make sure to include the following lines:

.. highlight:: yaml
.. code-block:: yaml

    hydra:
        searchpath:
            - file://${oc.env:CNEUROMAX_PATH}/cneuromax/

    defaults:
        - custom_hydra_zen_config_name
        - _self_
        - base_config

Replace the ``custom_hydra_zen_config_name`` accordingly.
Check-out the source code from
:mod:`~cneuromax.fitting.deeplearning.store_configs`
to see how/where this the name is defined (last line).

v. Create a ``__init__.py`` file in ``cneuromax/SERVICE_NAME/``
...............................................................

Example: ``cneuromax/fitting/deeplearning/__init__.py``
(`source file
<https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/fitting/deeplearning/__init__.py>`_)

.. warning::

    In order to be documented, all Python packages must contain a
    ``__init__.py`` file with at least one statement (``pass`` will do).

vi. Define a ``run`` function in ``__init__.py``
...............................................

This function details the execution of a ``task`` given a
:mod:`hydra-core` configuration. Should be of the form:

.. highlight:: python
.. code-block:: python

    @hydra.main(config_name="config",config_path=".",version_base=None)
    def run(config: DictConfig) -> None:
        ...

Check-out the source code from
:mod:`~cneuromax.fitting.deeplearning.run` for an example.

vii. Define a ``store_configs`` function in ``__init__.py``
.........................................................

This function registers the service's :mod:`hydra-core` configurations.
Should be of the form:

.. highlight:: python
.. code-block:: python

    def store_configs(cs: ConfigStore) -> None:
        ...

Check-out the source code from
:mod:`~cneuromax.fitting.deeplearning.store_configs` for an example.

3. Project
~~~~~~~~~~

a. Project overview
-------------------

A ``project`` is a single/multi-person endeavour with
a corresponding ``cneuromax/task/SERVICE_NAME/PROJECT_NAME/``
directory.

b. Example project
------------------

MNIST classification with ``fitting/deeplearning/classify_mnist``:

- :mod:`cneuromax.task.fitting.deeplearning.classify_mnist` \
(`source folder <https://github.com/\
courtois-neuromod/cneuromax/tree/main/\
cneuromax/task/fitting/deeplearning/classify_mnist>`_)

c. Creating new projects
------------------------

i. Create a ``cneuromax/task/SERVICE_NAME/PROJECT_NAME/`` directory
..................................................................

Example: ``cneuromax/task/fitting/deeplearning/classify_mnist``
(`source folder
<https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/task/fitting/deeplearning/classify_mnist>`_)

ii. Create a ``__init__.py`` file in the above directory
........................................................

Example:
``cneuromax/task/fitting/deeplearning/classify_mnist/__init__.py``
(`source file
<https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/task/fitting/deeplearning/classify_mnist/__init__.py>`_)

.. warning::

    In order to be documented, all Python packages must contain a
    ``__init__.py`` file with at least one statement (``pass`` will do).

iii. Define a ``store_configs`` function in ``__init__.py``
.........................................................

This function registers the project's :mod:`hydra-core` configurations.
Should be of the form:

.. highlight:: python
.. code-block:: python

    def store_configs(cs: ConfigStore) -> None:
        ...

Check-out the source code from
:mod:`~cneuromax.task.fitting.deeplearning.classify_mnist.store_configs`
for an example.

This function is meant to register the project's :mod:`hydra-core`
``task`` configurations to simplify your ``task`` config ``.yaml``
files by not having to specify the path to your classes with the
``_target_`` key.

4. Task
~~~~~~~

.. note::

    Two ambiguities to clarify:

    - a :mod:`cneuromax` ``task`` is not to be confused with a
      SLURM ``task`` which refers to the single or multi-process
      execution of a specific script.
    - throughout the documentation we use the term ``task execution``
      to refer to the process of running a ``task``. In contrast,
      we use ``execution`` to refer to the execution of a single
      script. Indeed, a ``task`` can consist in running a script
      multiple times, for instance when running hyperparameter
      optimization with both :mod:`hydra-core` and :mod:`orion`.

a. Task overview
----------------

A ``task`` is a work unit specified by a :mod:`hydra-core` configuration
``.yaml`` file located in a ``project`` subdirectory. A ``task``
with a corresponding
``cneuromax/task/SERVICE_NAME/PROJECT_NAME/TASK_NAME.yaml``
file can be executed by running

``python -m cneuromax task=SERVICE_NAME/PROJECT_NAME/TASK_NAME``.

.. note::

    Do not forget to:

    - prepend the Docker/Apptainer command or activate your virtual
      environment.
    - remove the ``.yaml`` extension at the end of the ``task``
      argument.

b. Example task
---------------

MLP MNIST classification with
``fitting/deeplearning/classify_mnist/mlp.yaml``
(`source file <https://github.com/\
courtois-neuromod/cneuromax/tree/main/\
cneuromax/task/fitting/deeplearning/classify_mnist/mlp.yaml>`_)

Usage:

``python -m cneuromax task=fitting/deeplearning/classify_mnist/mlp``

c. Creating new tasks
---------------------

i. Create ``cneuromax/task/SERVICE_NAME/PROJECT_NAME/TASK_NAME.yaml``
.................................................................

Example:
``cneuromax/task/fitting/deeplearning/classify_mnist/mlp_beluga.yaml``
(`source file <https://github.com/\
courtois-neuromod/cneuromax/tree/main/\
cneuromax/task/fitting/deeplearning/classify_mnist/mlp_beluga.yaml>`_)

Sphinx-incompatible files
=========================

``cneuromax/__main__.py``:

.. highlight:: python
.. code-block:: python

    from cneuromax import main

    if __name__ == "__main__":
        main()

``cneuromax/base_config.yaml``:

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
import warnings
from importlib import import_module

from beartype import BeartypeConf
from beartype.claw import beartype_this_package
from hydra.core.config_store import ConfigStore

from cneuromax.utils.run import (
    parse_task_argument,
    retrieve_module_functions,
)

beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))
warnings.filterwarnings("ignore", module="beartype")
warnings.filterwarnings("ignore", module="lightning")


def main() -> None:
    """Main :mod:`cneuromax` function called by ``__main__.py``.

    Parses the ``task`` command-line :mod:`hydra-core` argument,
    stores the corresponding ``service`` :mod:`hydra-core`
    configurations and runs the ``service`` which in turn
    runs the ``task`` with its :func:`hydra.main` decorated
    :func:`run` function.
    """
    service_name, project_name, task_name = parse_task_argument()
    store_module_configs, run_module = retrieve_module_functions(
        service_name=service_name,
    )
    cs = ConfigStore.instance()
    store_project_configs(cs)
    store_module_configs(cs)
    run_module()


def store_project_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` ``project`` configurations.

    Args:
        cs: A singleton instance that manages the :mod:`hydra-core`\
            configuration store.
    """
    service_name, project_name, _ = parse_task_argument()
    try:
        project_module = import_module(
            f"cneuromax.task.{service_name}.{project_name}",
        )
    except ModuleNotFoundError:
        logging.warning(
            f"The `__init__.py' file of project '{project_name}' cannot be"
            "found. Make sure it exists at "
            f"`cneuromax/task/{service_name}/{project_name}/__init__.py` and"
            "is spelled correctly. Check-out "
            "`cneuromax/task/fitting/deeplearning/classify_mnist/__init__.py`"
            "for an example.",
        )
        return
    try:
        project_module.store_configs(cs)
    except AttributeError:
        logging.exception(
            "The project module `store_configs` function cannot be found. Make"
            "sure it exists in your "
            f"`cneuromax/task/{service_name}/{project_name}/__init__.py` file."
            "Check-out "
            "`cneuromax/task/fitting/deeplearning/classify_mnist/__init__.py`"
            "for an example.",
        )
        return
