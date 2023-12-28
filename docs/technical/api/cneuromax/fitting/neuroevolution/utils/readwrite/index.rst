:py:mod:`cneuromax.fitting.neuroevolution.utils.readwrite`
==========================================================

.. py:module:: cneuromax.fitting.neuroevolution.utils.readwrite

.. autoapi-nested-parse::

   File reading and writing utilities for Neuroevolution fitting.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.utils.readwrite.load_state
   cneuromax.fitting.neuroevolution.utils.readwrite.save_state



.. py:function:: load_state(prev_num_gens: Annotated[int, ge(0)], len_agents_batch: Annotated[int, ge(1)]) -> tuple[list[list[cneuromax.fitting.neuroevolution.agent.singular.BaseSingularAgent]], cneuromax.fitting.neuroevolution.utils.type.generation_results_type | None, Annotated[int, ge(0)] | None]

   Load a previous experiment state from disk.

   :param prev_num_gens: See            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.prev_num_gens`.
   :param len_agents_batch: See return value of ``len_agents_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.

   :returns: See return value of ``agents_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
             generation_results: See return value of ``generation_results``            from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
             total_num_env_steps: See return value of            ``total_num_env_steps`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :rtype: agents_batch


.. py:function:: save_state(agents_batch: list[list[cneuromax.fitting.neuroevolution.agent.singular.BaseSingularAgent]], generation_results: cneuromax.fitting.neuroevolution.utils.type.generation_results_batch_type | None, total_num_env_steps: Annotated[int, ge(0)] | None, curr_gen: Annotated[int, ge(1)]) -> None

   Dump the current experiment state to disk.

   :param agents_batch: See return value of ``agents_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param generation_results: See return value of ``generation_results``            from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param total_num_env_steps: See return value of            ``total_num_env_steps`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param curr_gen: The current generation number.


