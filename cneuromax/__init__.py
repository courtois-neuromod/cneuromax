""":mod:`cneuromax` package.

Execution
=========

``python -m cneuromax project=PROJECT_NAME task=TASK_NAME``.

Terminology
===========

1. Quck definitions
~~~~~~~~~~~~~~~~~~~

``subtask``: Sub-work unit of a ``task`` (ex: a model training run
with a specific set of hyper-parameters).

``task``: Some work unit specified by a :mod:`hydra-core` config
``.yaml`` file or a :mod:`hydra-zen` Python config that specifies
its execution (ex: the training of the same type of model with various
hyper-parameters).

``project``: A collection of ``tasks`` + cross-``task``
functionality (ex: a custom :mod:`lightning` ``datamodule``).

``service``: Contains cross-``project`` functionality (ex: base
:mod:`lightning` sub-classes).

``interface``: Contains cross-``service`` functionality (ex:
:mod:`hydra-core` base configs).

2. Interface
~~~~~~~~~~~~

a. Interface overview
---------------------

An ``interface`` refers to a Python package located at
``cneuromax/INTERFACE_PATH/``.

.. note::

    Interfaces can be nested, ex: :mod:`cneuromax.serving`.

b. Example interfaces
---------------------

Root interface: :mod:`cneuromax` (`source folder <https://github.com/\
courtois-neuromod/cneuromax/tree/main/cneuromax/>`_)

Fitting: :mod:`cneuromax.fitting` (`source folder <https://github.com/\
courtois-neuromod/cneuromax/tree/main/cneuromax/fitting/>`_)

c. Creating a new interface
---------------------------

To create ``INTERFACE_NAME`` at path
``cneuromax/.../PARENT_INTERFACE_NAME/INTERFACE_NAME``, create a class
to inherit from the :class:`.BaseTaskRunner` class/sub-class implemented
by ``PARENT_INTERFACE_NAME`` (ex:
:class:`cneuromax.fitting.runner.FittingTaskRunner`).

3. Service
~~~~~~~~~~

a. Service overview
-------------------

A ``service`` refers to a Python package located at
``cneuromax/INTERFACE_PATH/SERVICE_NAME/``.

b. Example services
-------------------

Deep Learning: :mod:`cneuromax.fitting.deeplearning` (`source folder
<https://github.com/courtois-neuromod/cneuromax/tree/main/cneuromax/fitting/\
deeplearning>`_)

Neuroevolution: :mod:`cneuromax.fitting.neuroevolution` (`source folder
<https://github.com/courtois-neuromod/cneuromax/tree/main/cneuromax/fitting/\
neuroevolution>`_)

Model serving (in progress): :mod:`cneuromax.serving` (`source folder
<https://github.com/courtois-neuromod/cneuromax/tree/main/cneuromax/serving/>`_)

c. Creating a new service
-------------------------

To create ``SERVICE_NAME`` at path
``cneuromax/.../INTERFACE_LATEST_NAME/SERVICE_NAME``, create a class
to inherit from the :class:`.BaseTaskRunner` class/sub-class implemented
by ``INTERFACE_LATEST_NAME`` and implement as little as
:meth:`.BaseTaskRunner.run_subtask` (ex:
:class:`cneuromax.fitting.deeplearning.runner.DeepLearningTaskRunner`).

4. Project
~~~~~~~~~~

a. Project overview
-------------------

A ``project`` refers to a Python package located at
``cneuromax/projects/PROJECT_NAME/``.

b. Example projects
-------------------

MNIST classification: :mod:`cneuromax.projects.classify_mnist` (`source
folder <https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/projects/classify_mnist>`_)

Control tasks neuroevolution: :mod:`cneuromax.projects.control_nevo`
(`source folder <https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/projects/control_nevo>`_)

c. Creating a new project
-------------------------

To create ``PROJECT_NAME`` at path
``cneuromax/projects/PROJECT_NAME/``, create a class to inherit from
the :class:`.BaseTaskRunner` class/sub-class implemented by the
``service`` or other ``project`` of your choice (ex:
:class:`cneuromax.fitting.deeplearning.runner.DeepLearningTaskRunner`).
You probabaly will want to override
:meth:`~.BaseTaskRunner.store_configs`.

For succinctness (will reduce your command length), we suggest writing
the above class in the ``__init__.py`` file of your ``project``.

5. Task
~~~~~~~

a. Task overview
----------------

A ``task`` is a work unit specified by a :mod:`hydra-core` configuration
``.yaml`` file located in
``cneuromax/projects/PROJECT_NAME/task/TASK_NAME.yaml`` or a
:mod:`hydra-zen` Python config implemented in your overwritten
:meth:`.BaseTaskRunner.store_configs`.

b. Example tasks
----------------

MLP MNIST classification: ``cneuromax/projects/classify_mnist/task/\
mlp.yaml`` (`source file <https://github.com/courtois-neuromod/cneuromax/\
tree/main/cneuromax/projects/classify_mnist/task/mlp.yaml>`_)

Acrobot neuroevolution: Check out the contents of
:func:`cneuromax.projects.control_nevo.TaskRunner.store_configs`.

c. Creating a new task
----------------------

Create ``TASK_NAME.yaml`` at path
``cneuromax/projects/PROJECT_NAME/task/TASK_NAME.yaml`` and include
``# @package _global_`` at the top of the file (as shown
in the first above example). Otherwise, you can create a
:mod:`hydra-zen` Python config that specifies its execution (as shown
in the second above example).

__main__.py
===========

.. highlight:: python
.. code-block:: python

    from cneuromax.runner import BaseTaskRunner
    from cneuromax.utils.runner import get_task_runner_class
    from cneuromax.utils.wandb import login_wandb

    if __name__ == "__main__":
        TaskRunner: type[BaseTaskRunner] = get_task_runner_class()
        login_wandb()
        TaskRunner.store_configs_and_run_task()
"""
import os
import warnings

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

os.environ["OPENBLAS_NUM_THREADS"] = "1"
beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))
warnings.filterwarnings(action="ignore", module="beartype")
warnings.filterwarnings(action="ignore", module="lightning")
warnings.filterwarnings(action="ignore", module="gymnasium")
warnings.filterwarnings(action="ignore", module="torchrl")
