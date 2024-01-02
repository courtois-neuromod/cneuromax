""":class:`BaseAgent` & its config."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Annotated as An

from cneuromax.utils.annotations import ge, le


@dataclass(frozen=True)
class BaseAgentConfig:
    """Holds :class:`BaseAgent` config values.

    Args:

    """

    pass


class BaseAgent(metaclass=ABCMeta):
    """Root Neuroevolution agent class.

    From an algorithmic perspective, we make use of 50% truncation
    selection, meaning that the top 50% of agents in terms of fitness
    score are selected and will produce two children agents each.

    From an implementation perspective, ``pop_size`` instances of this
    class will be created upon initialization. Whenever an
    agent is selected, a copy of this object will be created and sent
    to a MPI process in possession of a non-selected agent. Both the
    original instance and the copy sent to the other process will be
    mutated in-place (meaning no new instance will be created).

    It might therefore be useful to sometimes consider this class as
    an ``AgentContainer`` class rather than an ``Agent`` class.

    Args:
        config: The instance's configuration.
        pop_idx: The agent's population index. An index of ``0`` means\
            that the agent is in the generator population while an\
            index of ``1`` means that the agent is in the\
            discriminator population.
        pops_are_merged: See\
            :paramref:`~.NeuroevolutionFittingHydraConfig.pop_merge`.

    Attributes:
        config (:class:`BaseAgentConfig`)
        role (``str``): The agent's role. Can be either ``"generator"``\
            or ``"discriminator"``.
        is_other_role_other_pop (``bool``): Whether the agent is the\
            other role in the other population. If the two populations\
            are merged (see :paramref:`pops_are_merged`), then an\
            agent is both a generator and a discriminator. It is a\
            generator/discriminator in this population while it is a\
            discriminator/generator in the other population. Such\
            type of agent needs to accomodate this property through\
            its network architecture.
        saved_env_state (``typing.Any``): The latest state of the\
            environment.
        saved_env_out (``tensordict.Tensordict``): The latest output\
            from the environment.
        saved_env_seed (``int``): The saved environment's seed.
        target_curr_episode_num_steps: (``int``):
        The target's current episode\
            number of steps. This attribute is only used if the\
            agent's :attr:`config`'s\
            :attr:`~.BaseAgentConfig.env_transfer` attribute is\
            ``True`` and the agent's :attr:`role` is\
            ``"discriminator"``.
        curr_episode_score: The current episode score. This attribute\
            is only used if the agent's :attr:`config`'s\
            :attr:`~.BaseAgentConfig.env_transfer` attribute is\
            ``True`` and the agent's :attr:`role` is\
            ``"generator"``.
        continual_fitness: The agent's continual fitness. This\
            attribute is only used if the agent's :attr:`config`'s\
            :attr:`~.BaseAgentConfig.fit_transfer` attribute is\
            ``True``.
    """

    def __init__(
        self: "BaseAgent",
        config: BaseAgentConfig,
        pop_idx: An[int, ge(0), le(1)],
        *,
        pops_are_merged: bool,
    ) -> None:
        self.config = config
        self.role = "generator" if pop_idx == 0 else "discriminator"

        self.is_other_role_other_pop = pops_are_merged
        self.initialize_evaluation_attributes()

    def initialize_evaluation_attributes(self: "BaseAgent") -> None:
        """Initializes attributes used during evaluation.

        If this agent's :attr:`role` is ``"discriminator"``, then
        all attributes
        """
        if self.config.env_transfer:
            self.saved_env_state = None
            self.saved_env_out = None
            self.saved_env_seed = None
            if self.role == "discriminator":
                self.target_curr_episode_num_steps = 0
            else:  # self.role == "generator"
                self.curr_episode_score = 0
        if self.config.fit_transfer:
            self.continual_fitness = 0

    @abstractmethod
    def mutate(self: "BaseAgent") -> None:
        """.

        Must be implemented.

        Args:
            seeds: An array of one or more random integers to seed the
                agent(s) mutation randomness.
        """
