:py:mod:`cneuromax.fitting.neuroevolution.space.base`
=====================================================

.. py:module:: cneuromax.fitting.neuroevolution.space.base

.. autoapi-nested-parse::

   :class:`BaseSpace` and its config class.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.space.base.BaseSpaceConfig
   cneuromax.fitting.neuroevolution.space.base.BaseSpaceAttrs
   cneuromax.fitting.neuroevolution.space.base.BaseSpace




.. py:class:: BaseSpaceConfig


   Holds :class:`BaseSpace` config values.

   :param eval_num_steps: Number of environment steps to run each agent            for during evaluation. `0` means that the agent will run            until the environment terminates (`eval_num_steps = 0` is            not supported for `env_transfer = True`).
   :param wandb_entity: See            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.wandb_entity`.

   .. py:attribute:: eval_num_steps
      :type: Annotated[int, ge(0)]

      


.. py:class:: BaseSpaceAttrs


   Holds :class:`BaseSpace`


.. py:class:: BaseSpace(config: BaseSpaceConfig)


   Spaces are virtual environments in which agents produce behaviour
   and receive fitness scores.

   .. py:property:: num_pops
      :type: Annotated[int, ge(1), le(2)]
      :abstractmethod:

      Number of agents interacting in a given space.

      As of now, there are two optimization paradigms for spaces:
      - Reinforcement Spaces, that only utilize 1 population of
          actor/generator agents.
      - Imitation Spaces, that utilize 2 populations of agents:
          actor/generator agents and discriminator agents.

      For a given evaluation, a "regular" space makes interact one
      agent from each population, whereas a batch space makes
      interact N agents from each population in parallel. See
      :class:`~BaseSpace.evaluates_on_gpu`.

   .. py:property:: evaluates_on_gpu
      :type: bool
      :abstractmethod:

      Whether this space evaluates agents on GPU or not.

      As of now, there are two execution paradigms for spaces:
      - CPU execution, where agents are evaluated sequentially on CPU.

   .. py:method:: evaluate(agent_s: list[list[cneuromax.fitting.neuroevolution.agent.singular.BaseSingularAgent]], curr_gen: Annotated[int, ge(1)]) -> numpy.typing.NDArray[numpy.float32]
      :abstractmethod:

      .

      Method called once per iteration (every generation) in order to
      evaluate and attribute fitnesses to agents.

      :param agent_s: Agent(s) to evaluate.
      :param curr_gen: Current generation.

      :returns:

                fitnesses and number of steps
                    ran.
      :rtype: The evaluation information



