:py:mod:`cneuromax.fitting.deeplearning.fit`
============================================

.. py:module:: cneuromax.fitting.deeplearning.fit

.. autoapi-nested-parse::

   Fitting function for Deep Learning.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.deeplearning.fit.fit



.. py:function:: fit(config: cneuromax.fitting.deeplearning.config.DeepLearningFittingHydraConfig) -> float

   Trains a Deep Learning model.

   This function is the main entry point of the Deep Learning module.
   It acts as an interface between :mod:`hydra-core` (configuration +
   launcher + sweeper) and :mod:`lightning` (trainer + logger +
   modules).

   Note that this function will be executed by
   ``num_nodes * gpus_per_node`` processes/tasks. Those variables are
   set in the Hydra launcher configuration.

   Trains (or resumes training) the model, saves a checkpoint and
   returns the final validation loss.

   :param config: The run's :mod:`hydra-core` structured config, see            :class:`~.DeepLearningFittingHydraConfig`.

   :returns: The final validation loss.


