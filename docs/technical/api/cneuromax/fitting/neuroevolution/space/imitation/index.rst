:py:mod:`cneuromax.fitting.neuroevolution.space.imitation`
==========================================================

.. py:module:: cneuromax.fitting.neuroevolution.space.imitation


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.space.imitation.BaseImitationTarget
   cneuromax.fitting.neuroevolution.space.imitation.BaseImitationSpace




.. py:class:: BaseImitationTarget


   Base Target class.

   .. py:method:: reset(seed: int, step_num: int) -> None
      :abstractmethod:

      Reset the target's state given a seed and step number.

      :param seed: Seed.
      :param step_num: Current step number.


   .. py:method:: __call__(x: torch.Tensor) -> torch.Tensor
      :abstractmethod:

      Inputs a value and returns an output.
      Reset the target's state

      :param x: Input value.

      :returns: Output value.
      :rtype: Any



.. py:class:: BaseImitationSpace(config: BaseSpaceConfig)




   Base Imitation Space class.

   Inside Imitation Spaces, agents evolve
   to imitate a target.

   .. py:property:: imitation_target
      :type: BaseImitationTarget
      :abstractmethod:

      The target to imitate.

   .. py:property:: hide_score
      :type: torch.Tensor

      Function that hides the environment's score portion of the screen
      to prevent the discriminator agent from utilizing it.

   .. py:property:: envs
      :type: torchrl.envs.EnvBase | tuple[torchrl.envs.EnvBase, torchrl.envs.EnvBase]
      :abstractmethod:

      One or two environments to run the generator and target.

   .. py:property:: init_reset
      :type: tensordict.TensorDict

      First reset function called during the match.
      Used to either set the env seed or resume from a previous state.

      :param curr_gen: Current generation.

      :returns: The initial environment observation.
      :rtype: np.ndarray

   .. py:method:: done_reset(curr_gen: int) -> numpy.ndarray

      Reset function called whenever the env returns done.

      :param curr_gen: Current generation.

      :returns: A new environment observation (np.ndarray).
      :rtype: np.ndarray


   .. py:method:: final_reset(obs: numpy.ndarray, curr_gen: Annotated[int, ge(1)]) -> None

      Reset function called at the end of every match.

      :param obs: The final environment observation.


   .. py:method:: evaluate(agent_s: list[list[cneuromax.fitting.neuroevolution.agent.singular.BaseSingularAgent]], curr_gen: Annotated[int, ge(1)]) -> numpy.ndarray

      .

      Method called once per iteration (every generation) in order to
      evaluate and attribute fitnesses to agents.

      :param agent_s: Agent(s) to evaluate.
      :param curr_gen: Current generation.

      :returns:

                fitnesses and number of steps
                    ran.
      :rtype: The evaluation information



