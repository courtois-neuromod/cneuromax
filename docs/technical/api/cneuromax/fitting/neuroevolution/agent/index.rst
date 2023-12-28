:py:mod:`cneuromax.fitting.neuroevolution.agent`
================================================

.. py:module:: cneuromax.fitting.neuroevolution.agent

.. autoapi-nested-parse::

   Neuroevolution Agents.



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   singular/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   base/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   cneuromax.fitting.neuroevolution.agent.BaseAgent




.. py:class:: BaseAgent(config: BaseAgentConfig, pop_idx: Annotated[int, ge(0), le(1)], *, pops_are_merged: bool)


   Root Neuroevolution agent class.

   From an algorithmic perspective, we make use of 50% truncation
   selection, meaning that the top 50% of agents in terms of fitness
   score are selected and will produce two children agents each.

   From an implementation perspective, ``pop_size`` instances of this
   class will be created upon the run's initialization. Whenever an
   agent is selected, a copy of this object will be created and sent
   to a MPI process in possession of a non-selected agent. Both the
   original instance and the copy sent to the other process will be
   mutated in-place (meaning no new instance will be created).

   It might therefore be useful to sometimes consider this class as
   an ``AgentContainer`` class rather than an ``Agent`` class.

   :param config: See :class:`BaseAgentConfig`.
   :param pop_idx: The agent's population index. An index of ``0`` means            that the agent is in the generator population while an            index of ``1`` means that the agent is in the            discriminator population.
   :param pops_are_merged: See            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.pop_merge`.

   .. attribute:: config

      

      :type: :class:`BaseAgentConfig`

   .. attribute:: role

      The agent's role. Can be either ``"generator"``            or ``"discriminator"``.

      :type: ``str``

   .. attribute:: is_other_role_other_pop

      Whether the agent is the            other role in the other population. If the two populations            are merged (see :paramref:`pops_are_merged`), then an            agent is both a generator and a discriminator. It is a            generator/discriminator in this population while it is a            discriminator/generator in the other population. Such            type of agent needs to accomodate this property through            its network architecture.

      :type: ``bool``

   .. attribute:: saved_env_state

      The latest state of the            environment.

      :type: ``typing.Any``

   .. attribute:: saved_env_out

      The latest output            from the environment.

      :type: ``tensordict.Tensordict``

   .. attribute:: saved_env_seed

      The saved environment's seed.

      :type: ``int``

   .. attribute:: target_curr_episode_num_steps

      (``int``):

   .. attribute:: The target's current episode            number of steps. This attribute is only used if the            agent's :attr:`config`'s            :attr:`~.BaseAgentConfig.env_transfer` attribute is            ``True`` and the agent's :attr:`role` is            ``"discriminator"``.

      

   .. attribute:: curr_episode_score

      The current episode score. This attribute            is only used if the agent's :attr:`config`'s            :attr:`~.BaseAgentConfig.env_transfer` attribute is            ``True`` and the agent's :attr:`role` is            ``"generator"``.

   .. attribute:: continual_fitness

      The agent's continual fitness. This            attribute is only used if the agent's :attr:`config`'s            :attr:`~.BaseAgentConfig.fit_transfer` attribute is            ``True``.

   .. py:method:: initialize_evaluation_attributes() -> None

      Initializes attributes used during evaluation.

      If this agent's :attr:`role` is ``"discriminator"``, then
      all attributes


   .. py:method:: mutate() -> None
      :abstractmethod:

      .

      Must be implemented.

      :param seeds: An array of one or more random integers to seed the
                    agent(s) mutation randomness.



