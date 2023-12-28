:py:mod:`cneuromax.fitting.hybrid`
==================================

.. py:module:: cneuromax.fitting.hybrid

.. autoapi-nested-parse::

   Fitting w/ Hybrid DL & NE (+ :mod:`hydra-core` config storing).



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   __main__/index.rst
   config/index.rst
   fit/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.hybrid.HybridFittingHydraConfig



Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.hybrid.store_task_configs
   cneuromax.fitting.hybrid.store_base_fitting_configs
   cneuromax.fitting.hybrid.store_hybrid_fitting_configs



.. py:function:: store_task_configs(cs: hydra.core.config_store.ConfigStore) -> None

   Stores :mod:`hydra-core` task configurations.

   Parses the task config path from the script arguments, import
   its ``store_configs`` function if it exists, and calls it.

   :param cs: A singleton instance that manages the :mod:`hydra-core`            configuration store.

   :raises ModuleNotFoundError: If the task module cannot be found.
   :raises AttributeError: If the task module does not have a            ``store_configs`` function.


.. py:function:: store_base_fitting_configs(cs: hydra.core.config_store.ConfigStore) -> None

   Stores :mod:`hydra-core` fitting configs.

   :param cs: See :paramref:`~cneuromax.__init__.store_task_configs.cs`.


.. py:class:: HybridFittingHydraConfig




   Structured :mod:`hydra-core` config for DL + NE fitting.


.. py:function:: store_hybrid_fitting_configs() -> None

   Stores :mod:`hydra-core` Hybrid DL + NE fitting configs.


