""":mod:`cneuromax` package.

Terminology
==============

1. Quck definitions
~~~~~~~~~~~~~~~~~~~

**Task**: Some work unit specified by a :mod:`hydra-core` config
``.yaml`` file or a :mod:`hydra-zen` Python function that specifies
its execution.

**Subtask**: Some pure Python sub-work unit of a ``task``.

**Project**: A collection of ``tasks`` + cross-``task``
functionality (ex: a custom :mod:`lightning` ``datamodule``).

**Service**: Contains cross-``project`` functionality (ex: base
:mod:`lightning` sub-classes).

**Interface**: Contains cross-``service`` functionality (ex: the
:mod:`hydra-core` base configs).

2. Interface
~~~~~~~~~~~~

a. Interface overview
---------------------

An ``interface`` refers to a Python package located at
``cneuromax/INTERFACE_PATH/``.

.. note::

    Interfaces can be nested, ex: :mod:`cneuromax.fitting`.

b. Example interfaces
---------------------

Base interface: :mod:`cneuromax` (`source folder <https://github.com/\
courtois-neuromod/cneuromax/tree/main/cneuromax/>`_)

Fitting: :mod:`cneuromax.fitting` (`source folder <https://github.com/\
courtois-neuromod/cneuromax/tree/main/cneuromax/fitting/>`_)

c. Creating a new interface
---------------------------

To create ``INTERFACE_NAME`` at path
``cneuromax/.../PARENT_INTERFACE_NAME/INTERFACE_NAME``, create a class
to inherit from the :class:`.BaseTaskRunner` class/sub-class implemented
by ``PARENT_INTERFACE_NAME``. Feel free to also override
:attr:`.BaseTaskRunner.hydra_config` & implement
:meth:`.BaseTaskRunner.store_configs`.

.. warning::

    Make sure to call the parent method if your ``interface`` is nested.

Example: :class:`cneuromax.fitting.runner.FittingTaskRunner`.

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
``cneuromax/.../LATEST_INTERFACE_NAME/SERVICE_NAME``, create a class
to inherit from the :class:`.BaseTaskRunner` class/sub-class implemented
by ``LATEST_INTERFACE_NAME`` and implement as little as
:meth:`.BaseTaskRunner.run_subtask`. Feel free to also override
:attr:`.BaseTaskRunner.hydra_config` & implement
:meth:`.BaseTaskRunner.store_configs`.

Example:\
 :class:`cneuromax.fitting.deeplearning.runner.DeepLearningTaskRunner`.

4. Project
~~~~~~~~~~

a. Project overview
-------------------

A ``project`` refers to a Python package located at
``cneuromax/projects/PROJECT_NAME/``.

b. Example project
------------------

MNIST classification: :mod:`cneuromax.projects.classify_mnist` (`source
folder <https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/projects/classify_mnist>`_)

c. Creating a new project
-------------------------

To create ``PROJECT_NAME`` at path
``cneuromax/projects/PROJECT_NAME/``, create a class in ``__main__.py``
to inherit from the :class:`.BaseTaskRunner` class/sub-class implemented
by the ``service`` of your choice. Feel free to override
:attr:`.BaseTaskRunner.hydra_config` & implement
:meth:`.BaseTaskRunner.store_configs`.

Finally make sure to add a ``__main__.py`` file to your ``project``
directory as ``projects`` are the entrypoint to executing ``tasks``.

Check-out ``cneuromax.projects.classify_mnist.__main__.py`` as an
`example <https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/projects/classify_mnist/__main__.py>`_.

5. Task
~~~~~~~

a. Task overview
----------------

A ``task`` is a work unit specified by a :mod:`hydra-core` configuration
``.yaml`` file located in
``cneuromax/projects/PROJECT_NAME/task/TASK_NAME.yaml``.

Can be executed with the following command:

``python -m cneuromax.projects.PROJECT_NAME task=TASK_NAME``.

b. Example task
---------------

MLP MNIST classification: ``cneuromax/projects/classify_mnist/task/\
mlp.yaml`` (`source file <https://github.com/courtois-neuromod/cneuromax/\
tree/main/cneuromax/projects/classify_mnist/task/mlp.yaml>`_)

c. Creating a new task
----------------------

Create ``TASK_NAME.yaml`` at path
``cneuromax/projects/PROJECT_NAME/task/TASK_NAME.yaml`` and include
as little as ``# @package _global_`` at the top of the file.
"""
import warnings

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))
warnings.filterwarnings("ignore", module="beartype")
warnings.filterwarnings("ignore", module="lightning")
