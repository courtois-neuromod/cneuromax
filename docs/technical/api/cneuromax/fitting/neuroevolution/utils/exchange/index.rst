:py:mod:`cneuromax.fitting.neuroevolution.utils.exchange`
=========================================================

.. py:module:: cneuromax.fitting.neuroevolution.utils.exchange

.. autoapi-nested-parse::

   Process agent exchange for Neuroevolution fitting.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.utils.exchange.exchange_agents
   cneuromax.fitting.neuroevolution.utils.exchange.update_exchange_and_mutate_info



.. py:function:: exchange_agents(num_pops: Annotated[int, ge(1), le(2)], pop_size: Annotated[int, ge(1)], agents_batch: list[list[cneuromax.fitting.neuroevolution.agent.singular.BaseSingularAgent]], exchange_and_mutate_info_batch: cneuromax.fitting.neuroevolution.utils.type.exchange_and_mutate_info_batch_type) -> None

   Exchange agents between processes.

   :param num_pops: See            :meth:`~.neuroevolution.space.base.BaseSpace.num_pops`.
   :param pop_size: See return value of ``pop_size`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param agents_batch: See return value of ``agents_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param exchange_and_mutate_info_batch: See return value of            ``exchange_and_mutate_info_batch`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.


.. py:function:: update_exchange_and_mutate_info(num_pops: Annotated[int, ge(1), le(2)], pop_size: Annotated[int, ge(1)], exchange_and_mutate_info: cneuromax.fitting.neuroevolution.utils.type.exchange_and_mutate_info_type | None, generation_results: cneuromax.fitting.neuroevolution.utils.type.generation_results_type | None, seeds: cneuromax.fitting.neuroevolution.utils.type.seeds_type | None) -> None

   Update the exchange and mutate information.

   The selection process of the algorithm is in some sense implicit in    `cneuromax`. We make use of 50% truncation selection, which is    reflected in the information stored inside
   :paramref:`exchange_and_mutate_info`.

   In some sense, the selection process of the algorithm is performed
   in this function.

   :param num_pops: See            :meth:`~.neuroevolution.space.base.BaseSpace.num_pops`.
   :param pop_size: See return value of ``pop_size`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param exchange_and_mutate_info: See return value of            ``exchange_and_mutate_info`` from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param generation_results: See return value of ``generation_results``            from            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
   :param seeds: See return value of ``seeds`` from            :func:`~.neuroevolution.utils.compute.compute_start_time_and_seeds`.


