:py:mod:`cneuromax.fitting.neuroevolution.fit`
==============================================

.. py:module:: cneuromax.fitting.neuroevolution.fit

.. autoapi-nested-parse::

   Neuroevolution fitting.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.fit.fit



.. py:function:: fit(config: cneuromax.fitting.neuroevolution.config.NeuroevolutionFittingHydraConfig) -> None

   Fitting function for Neuroevolution algorithms.

   This function is the main entry point of the Neuroevolution module.
   It acts as an interface between Hydra (configuration + launcher +
   sweeper), Spaces, Agents and MPI resource scheduling for
   Neuroevolution algorithms.

   Note that this function and all of its sub-functions will be called
   by `num_nodes * tasks_per_node` MPI processes/tasks. These two
   variables are set in the Hydra launcher configuration.

   :param config: The run's :mod:`hydra-core` structured config, see            :class:`~.NeuroevolutionFittingHydraConfig`.


