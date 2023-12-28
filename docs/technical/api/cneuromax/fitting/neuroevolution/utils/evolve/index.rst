:py:mod:`cneuromax.fitting.neuroevolution.utils.evolve`
=======================================================

.. py:module:: cneuromax.fitting.neuroevolution.utils.evolve

.. autoapi-nested-parse::

   Evolutionary operations for Neuroevolution fitting.

   The selection operation is implicit in `cneuromax`, see
   :func:`~.neuroevolution.utils.exchange.update_exchange_and_mutate_info`
   for more details.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.utils.evolve.run_mutation
   cneuromax.fitting.neuroevolution.utils.evolve.run_evaluation_cpu
   cneuromax.fitting.neuroevolution.utils.evolve.run_evaluation_gpu



.. py:function:: run_mutation(agents_batch: list[list[cneuromax.fitting.neuroevolution.agent.singular.BaseSingularAgent]], exchange_and_mutate_info_batch: cneuromax.fitting.neuroevolution.utils.type.exchange_and_mutate_info_batch_type, num_pops: int) -> None

   Mutate all agents maintained by this process.

   :param agents_batch: See return value of ``agents_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param exchange_and_mutate_info_batch: See return value of            ``exchange_and_mutate_info_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param num_pops: See            :meth:`~.neuroevolution.space.base.BaseSpace.num_pops`.


.. py:function:: run_evaluation_cpu(agents_batch: list[list[cneuromax.fitting.neuroevolution.agent.singular.BaseSingularAgent]], space: cneuromax.fitting.neuroevolution.space.base.BaseSpace, curr_gen: Annotated[int, ge(1)]) -> cneuromax.fitting.neuroevolution.utils.type.fitnesses_and_num_env_steps_batch_type

   Evaluate all agents maintained by this process.

   :param agents_batch: See return value of ``agents_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param space: The experiment's pace, see :class:`~.BaseSpace`.
   :param curr_gen: Current generation number.

   :returns: The output of            agent evaluation by this process. See            :meth:`~.BaseSpace.evaluate`.
   :rtype: fitnesses_and_num_env_steps_batch


.. py:function:: run_evaluation_gpu(ith_gpu_comm: mpi4py.MPI.Comm, agents_batch: list[list[cneuromax.fitting.neuroevolution.agent.singular.BaseSingularAgent]], space: cneuromax.fitting.neuroevolution.space.base.BaseSpace, curr_gen: Annotated[int, ge(1)], *, transfer: bool) -> cneuromax.fitting.neuroevolution.utils.type.fitnesses_and_num_env_steps_batch_type

   Evaluate all agents maintained by this process.

   :param ith_gpu_comm: See return value of ``ith_gpu_comm`` from            :func:`~.neuroevolution.utils.initialize.initialize_gpu_comm`.
   :param agents_batch: See return value of ``agents_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param space: The experiment's pace, see :class:`~.BaseSpace`.
   :param curr_gen: Current generation number.
   :param transfer: Whether any of            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.env_transfer`,            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.fit_transfer`            or            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.mem_transfer`            is `True`.

   :returns: The output of            agent evaluation by this process. See            :meth:`~.BaseSpace.evaluate`.
   :rtype: fitnesses_and_num_env_steps_batch


