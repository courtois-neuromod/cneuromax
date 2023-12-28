:py:mod:`cneuromax.fitting.neuroevolution.utils.compute`
========================================================

.. py:module:: cneuromax.fitting.neuroevolution.utils.compute

.. autoapi-nested-parse::

   Functions requiring some computation for Neuroevolution fitting.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.utils.compute.compute_generation_results
   cneuromax.fitting.neuroevolution.utils.compute.compute_save_points
   cneuromax.fitting.neuroevolution.utils.compute.compute_start_time_and_seeds
   cneuromax.fitting.neuroevolution.utils.compute.compute_total_num_env_steps_and_process_fitnesses



.. py:function:: compute_generation_results(generation_results: cneuromax.fitting.neuroevolution.utils.type.generation_results_type | None, generation_results_batch: cneuromax.fitting.neuroevolution.utils.type.generation_results_batch_type, fitnesses_and_num_env_steps_batch: cneuromax.fitting.neuroevolution.utils.type.fitnesses_and_num_env_steps_batch_type, agents_batch: list[list[cneuromax.fitting.neuroevolution.agent.singular.BaseSingularAgent]], num_pops: Annotated[int, ge(1)]) -> None

   Computes generation results across processes and gathers them.

   :param generation_results: See return value of ``generation_results``            from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param generation_results_batch: See return value of            ``generation_results_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param fitnesses_and_num_env_steps_batch: See return value of            ``fitnesses_and_num_env_steps_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param agents_batch: See return value of ``agents_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param num_pops: See            :meth:`~.neuroevolution.space.base.BaseSpace.num_pops`.


.. py:function:: compute_save_points(prev_num_gens: Annotated[int, ge(0)], total_num_gens: Annotated[int, ge(0)], save_interval: Annotated[int, ge(0)], *, save_first_gen: bool) -> list[int]

   Compute generations at which to save the state.

   :param prev_num_gens: See            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.prev_num_gens`.
   :param total_num_gens: See            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.total_num_gens`.
   :param save_interval: See            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.save_interval`.
   :param save_first_gen: See            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.save_first_gen`.

   :returns: A list of generations at which to save the state.


.. py:function:: compute_start_time_and_seeds(generation_results: cneuromax.fitting.neuroevolution.utils.type.generation_results_type | None, curr_gen: Annotated[int, ge(1)], num_pops: Annotated[int, ge(1)], pop_size: Annotated[int, ge(1)], *, pop_merge: bool) -> tuple[float | None, cneuromax.fitting.neuroevolution.utils.type.seeds_type | None]

   Compute the start time and seeds for the current generation.

   :param generation_results: See return value of ``generation_results``            from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param curr_gen: Current generation number.
   :param num_pops: See            :meth:`~.neuroevolution.space.base.BaseSpace.num_pops`.
   :param pop_size: See return value of ``pop_size`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param pop_merge: See            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.pop_merge`.

   :returns:

             * **start_time** - The start time for the current generation.
             * **seeds** - The seeds for the current generation.


.. py:function:: compute_total_num_env_steps_and_process_fitnesses(generation_results: cneuromax.fitting.neuroevolution.utils.type.generation_results_type | None, total_num_env_steps: Annotated[int, ge(0)] | None, curr_gen: Annotated[int, ge(1)], start_time: float | None, *, pop_merge: bool) -> Annotated[int, ge(0)] | None

   Processes the generation results.

   :param generation_results: See return value of ``generation_results``            from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param total_num_env_steps: See return value of            ``total_num_env_steps`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param curr_gen: Current generation number.
   :param start_time: Generation start time.
   :param pop_merge: See            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.pop_merge`.

   :returns: The updated total number of environment steps.


