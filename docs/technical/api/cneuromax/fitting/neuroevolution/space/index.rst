:py:mod:`cneuromax.fitting.neuroevolution.space`
================================================

.. py:module:: cneuromax.fitting.neuroevolution.space

.. autoapi-nested-parse::

   Neuroevolution Spaces.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   base/index.rst
   batch_imitation/index.rst
   batch_reinforcement/index.rst
   imitation/index.rst
   reinforcement/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.space.BaseSpace
   cneuromax.fitting.neuroevolution.space.BaseImitationSpace
   cneuromax.fitting.neuroevolution.space.BaseImitationTarget
   cneuromax.fitting.neuroevolution.space.BaseReinforcementSpace




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



