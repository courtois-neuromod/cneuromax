:py:mod:`cneuromax.fitting.neuroevolution.utils.validate`
=========================================================

.. py:module:: cneuromax.fitting.neuroevolution.utils.validate

.. autoapi-nested-parse::

   Run validation for Neuroevolution fitting.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.utils.validate.validate_space



.. py:function:: validate_space(space: cneuromax.fitting.neuroevolution.space.base.BaseSpace, *, pop_merge: bool) -> None

   Makes sure that the run's Space is valid given the run's config.

   :param space: The run's :class:`~.BaseSpace` instance.
   :param pop_merge: See            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.pop_merge`.


