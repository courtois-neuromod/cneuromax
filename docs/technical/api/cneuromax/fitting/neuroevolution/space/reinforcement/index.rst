:py:mod:`cneuromax.fitting.neuroevolution.space.reinforcement`
==============================================================

.. py:module:: cneuromax.fitting.neuroevolution.space.reinforcement

.. autoapi-nested-parse::

   .



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.space.reinforcement.BaseReinforcementSpace




.. py:class:: BaseReinforcementSpace(config: BaseSpaceConfig)




   Base Reinforcement Space class.

   Inside Reinforcement Spaces, agents evolve to maximize a reward
   function.

   .. py:property:: env
      :type: torchrl.envs.EnvBase
      :abstractmethod:

      Environment to run the agent.

   .. py:property:: num_pops
      :type: int

      See :class:`~.BaseSpace.num_pops`.

   .. py:property:: evaluates_on_gpu
      :type: bool
      :abstractmethod:

      See :class:`~.BaseSpace.evaluates_on_gpu`.

   .. py:method:: init_reset(curr_gen: int) -> tensordict.TensorDict

      First reset function called during the run.

      Used to reset the
      environment & potentially resume from a previous state.

      :param curr_gen: Current generation.

      :returns: The initial environment observation.
      :rtype: np.ndarray


   .. py:method:: done_reset(curr_gen: int) -> tensordict.TensorDict

      Reset function called whenever the environment returns done.

      :param curr_gen: Current generation.

      :returns: A new environment observation (np.ndarray).
      :rtype: np.ndarray


   .. py:method:: final_reset(out: tensordict.TensorDict, curr_gen: int) -> None

      Reset function called at the end of every run.

      :param obs: The final environment observation.


   .. py:method:: evaluate(agent_s: list[list[cneuromax.fitting.neuroevolution.agent.singular.BaseSingularAgent]], curr_gen: Annotated[int, ge(1)]) -> float

      .



