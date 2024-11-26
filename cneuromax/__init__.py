""":mod:`cneuromax` package.

Keywords
========

``interface``, ``service``, ``project``, ``schedule``, ``task``, ``run``

Structure
=========

```
cneuromax/
├─ INTERFACE_NAME_1/ # ex: fitting
│  ├─ SERVICE_NAME_1/ # ex: deeplearning
│  │  └─ ...
│  └─ ...
├─ ...
└─ projects/
   ├─ PROJECT_NAME_1/ # ex: classify_mnist
   │  ├─ schedule/
   │  │  ├─ SCHEDULE_NAME_1.yaml
   │  │  └─ ...
   │  └─ task/
   │     ├─ TASK_NAME_1.yaml # ex: mlp mnist classification
   │     └─ ...
   └─ ...
```

Examples
========

- ``project``

MNIST classification: :mod:`cneuromax.projects.classify_mnist` (`source
folder <https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/projects/classify_mnist>`_)

Reinforcement Learning on Control Tasks w/ Neuroevolution:
:mod:`cneuromax.projects.neuroevo_rl_control`
(`source folder <https://github.com/courtois-neuromod/cneuromax/tree/main/\
cneuromax/projects/neuroevo_rl_control>`_)

- ``task``

MLP MNIST classification: ``cneuromax/projects/classify_mnist/task/\
mlp.yaml`` (`source file <https://github.com/courtois-neuromod/cneuromax/\
tree/main/cneuromax/projects/classify_mnist/task/mlp.yaml>`_)

Execution
=========

Running a ``task``:

``python -m cneuromax project=PROJECT_NAME task=TASK_NAME``.

Running a ``schedule`` (a sequence of ``tasks``):

``python -m cneuromax project=PROJECT_NAME schedule=SCHEDULE_NAME``.

I. Overview & Examples
~~~~~~~~~~~~~~~~~~~~~~

- ``project``



- ``task``



1. ``project``
~~~~~~~~~~~~~~

a. Examples
-----------



Contribution
============

To create ``PROJECT_NAME`` at path
``cneuromax/projects/PROJECT_NAME/``, create a class to inherit from
the :class:`.BaseTaskRunner` class/sub-class implemented by the
``service`` or other ``project`` of your choice (ex:
:class:`cneuromax.fitting.deeplearning.runner.DeepLearningTaskRunner`).
You probabaly will want to override
:meth:`~.BaseTaskRunner.store_configs`.

For succinctness (will reduce your command length), we suggest writing
the above class in the ``__init__.py`` file of your ``project``.


``run``: Sub-work unit of a ``task`` (ex: a model training run
with a specific set of hyper-parameters).

``task``: Some work unit specified by a `Hydra <https://hydra.cc>`_
``.yaml`` or a :doc:`hydra-zen <hydra-zen:index>` Python config that
specifies its execution (ex: the training of the same type of model
with various hyper-parameters).

``scheduler``:

``project``: A collection of ``tasks`` + cross-``task``
functionality (ex: a custom
:class:`lightning.pytorch.core.LightningDataModule`)

``service``: Contains cross-``project`` functionality (ex: base
`Lightning <https://lightning.ai/>`_ sub-classes).

``interface``: Contains cross-``service`` functionality (ex:
`Hydra <https://hydra.cc>`_ base configs).

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
:meth:`.BaseTaskRunner.run` (ex:
:class:`cneuromax.fitting.deeplearning.runner.DeepLearningTaskRunner`).



5. Task
~~~~~~~

a. Task overview
----------------

A ``task`` is a work unit specified by a `Hydra <https://hydra.cc>`_
configuration ``.yaml`` file located in
``cneuromax/projects/PROJECT_NAME/task/TASK_NAME.yaml`` or a
:doc:`hydra-zen <hydra-zen:index>` Python config implemented in your
overwritten :meth:`.BaseTaskRunner.store_configs`.

b. Example tasks
----------------



Acrobot neuroevolution: Check out the contents of
:func:`cneuromax.projects.control_nevo.TaskRunner.store_configs`.

c. Creating a new task
----------------------

Create ``TASK_NAME.yaml`` at path
``cneuromax/projects/PROJECT_NAME/task/TASK_NAME.yaml`` and include
``# @package _global_`` at the top of the file (as shown
in the first above example). Otherwise, you can create a
:doc:`hydra-zen <hydra-zen:index>` Python config that specifies its
execution (as shown in the second above example).

__main__.py
===========

.. highlight:: python
.. code-block:: python

    from omegaconf import OmegaConf

    from cneuromax.runner import BaseTaskRunner
    from cneuromax.utils.runner import get_task_runner_class
    from cneuromax.utils.wandb import login_wandb

    if __name__ == "__main__":
        TaskRunner: type[BaseTaskRunner] = get_task_runner_class()
        OmegaConf.register_new_resolver("eval", eval)
        login_wandb()
        TaskRunner.store_configs_and_start_runs()
"""

from collections.abc import Callable
from typing import Any

from beartype import BeartypeConf
from beartype.claw import beartype_this_package
from hydra_zen import ZenStore
from lightning.pytorch.loggers.wandb import WandbLogger

from cneuromax.utils.hydra_zen import pfs_builds

beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))


def store_wandb_logger_configs(
    store: ZenStore,
    clb: Callable[..., Any],
) -> None:
    """Stores `Hydra <https://hydra.cc>`_ ``logger`` group configs.

    Config names: ``wandb``, ``wandb_simexp``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        clb: `W&B <https://wandb.ai/>`_ initialization callable.
    """
    dir_key = "save_dir" if clb == WandbLogger else "dir"
    base_args: dict[str, Any] = {  # `fs_builds`` does not like dict[str, str]
        "name": "${schedule}/${task}/${hydra:job.override_dirname}",
        dir_key: "${hydra:sweep.dir}/${now:%Y-%m-%d-%H-%M-%S}",
        "project": "${project}",
    }
    store(
        pfs_builds(clb, **base_args),
        group="logger",
        name="wandb",
    )
    store(
        pfs_builds(clb, **base_args, entity="cneuroml"),
        group="logger",
        name="wandb_simexp",
    )
