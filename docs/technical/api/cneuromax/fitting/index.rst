:py:mod:`cneuromax.fitting`
===========================

.. py:module:: cneuromax.fitting

.. autoapi-nested-parse::

   Fitting module (+ :mod:`hydra-core` config storing).



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   deeplearning/index.rst
   hybrid/index.rst
   neuroevolution/index.rst


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

   cneuromax.fitting.store_base_fitting_configs



.. py:function:: store_base_fitting_configs(cs: hydra.core.config_store.ConfigStore) -> None

   Stores :mod:`hydra-core` fitting configs.

   :param cs: See :paramref:`~cneuromax.__init__.store_task_configs.cs`.


