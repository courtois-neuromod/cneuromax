""":class:`BaseAgent` & its config."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Annotated as An

from torch import Tensor

from cneuromax.utils.beartype import ge, le


@dataclass
class BaseAgentConfig:
    """Holds :class:`BaseAgent` config values.

    Args:
        env_transfer: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.env_transfer`.
        fit_transfer: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.fit_transfer`.
        mem_transfer: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.mem_transfer`.
    """

    env_transfer: bool = "${config.env_transfer}"  # type: ignore[assignment]
    fit_transfer: bool = "${config.fit_transfer}"  # type: ignore[assignment]
    mem_transfer: bool = "${config.mem_transfer}"  # type: ignore[assignment]


class BaseAgent(metaclass=ABCMeta):
    """Root Neuroevolution agent class.

    From an algorithmic perspective, we make use of 50% truncation
    selection, meaning that the top 50% of agents in terms of fitness
    score are selected and will produce two children agents each.

    From an implementation perspective, ``pop_size`` instances of this
    class will be created upon initialization. Whenever an
    agent is selected, a copy of this object will be created and sent
    to a MPI process in possession of a non-selected agent. Both this
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
            :paramref:`~.NeuroevolutionSubtaskConfig.pop_merge`.

    Attributes:
        config (:class:`BaseAgentConfig`)
        role (``str``): The agent's role. Can be either ``"generator"``\
            or ``"discriminator"``.
        is_other_role_in_other_pop (``bool``): Whether the agent is the\
            other role in the other population. If the two populations\
            are merged (see :paramref:`pops_are_merged`), then an\
            agent is both a generator and a discriminator. It is a\
            generator/discriminator in this population while it is a\
            discriminator/generator in the other population. Such\
            type of agent needs to accomodate this property through\
            its network architecture.
        curr_eval_score (``float``): The score obtained by the agent\
            during the current evaluation.
        curr_eval_num_steps (``int``): The number of steps taken by the\
            agent during the current evaluation.
        saved_env (``torchrl.envs.EnvBase``): The :mod:`torchrl`\
            environment instance to resume from (only set if
            :paramref:`~.BaseAgentConfig.env_transfer` is ``True``).
        saved_env_out (``tensordict.Tensordict``): The latest output\
            from the environment to resume from (only set if\
            :paramref:`~.BaseAgentConfig.env_transfer` is ``True``).
        curr_episode_score: The current episode score (only set if\
            :paramref:`~.BaseAgentConfig.env_transfer` is ``True``).
        curr_episode_num_steps: The number of steps taken in the\
            current episode (only set if\
            :paramref:`~.BaseAgentConfig.env_transfer` is ``True``).
        continual_fitness: The agent's fitness in addition to all of\
            its predecessors' fitnesses (only set if\
            :paramref:`~.BaseAgentConfig.fit_transfer` is ``True``).
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
        self.is_other_role_in_other_pop = pops_are_merged
        self.initialize_eval_attributes()

    def initialize_eval_attributes(self: "BaseAgent") -> None:
        """Initializes attributes used during evaluation."""
        self.curr_eval_score = 0
        self.curr_eval_num_steps = 0
        if self.config.env_transfer:
            self.saved_env = None
            self.saved_env_out = None
            self.curr_episode_num_steps = 0
            self.curr_episode_score = 0
        if self.config.fit_transfer:
            self.continual_fitness = 0

    @property
    def seed(self: "BaseAgent") -> int:
        """The agent's seed used to fix its randomness."""
        return self._seed

    @seed.setter
    def seed(self: "BaseAgent", seed: int) -> None:
        self._seed = seed

    @abstractmethod
    def mutate(self: "BaseAgent") -> None:
        """Applies random mutation(s) to the agent."""

    @abstractmethod
    def reset(self: "BaseAgent") -> None:
        """Resets the agent's memory state."""

    @abstractmethod
    def __call__(self: "BaseAgent", x: Tensor) -> Tensor:
        """Runs the agent for one timestep given :paramref:`x`.

        Args:
            x: An input observation.

        Returns:
            The agent's output.
        """
