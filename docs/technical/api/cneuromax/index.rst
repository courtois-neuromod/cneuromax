:py:mod:`cneuromax`
===================

.. py:module:: cneuromax

.. autoapi-nested-parse::

   :mod:`cneuromax` codebase (+ store :mod:`hydra-core` task config).



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   fitting/index.rst
   serving/index.rst
   task/index.rst
   utils/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   config/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.store_task_configs



.. py:function:: store_task_configs(cs: hydra.core.config_store.ConfigStore) -> None

   Stores :mod:`hydra-core` task configurations.

   Parses the task config path from the script arguments, import
   its ``store_configs`` function if it exists, and calls it.

   :param cs: A singleton instance that manages the :mod:`hydra-core`            configuration store.

   :raises ModuleNotFoundError: If the task module cannot be found.
   :raises AttributeError: If the task module does not have a            ``store_configs`` function.


