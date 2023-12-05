"""."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseAgentConfig:
    """Holds :class:`BaseAgent` config values.

    Args:
        env_transfer: Whether the agent's environment state\
            (position, velocity, ...) is transferred to its children.
        fit_transfer: Whether the agent's fitness is transferred to\
            its children.
        mem_transfer: Whether the agent's hidden state/memory is\
            transferred to its children.
    """

    env_transfer: bool = False
    fit_transfer: bool = False
    mem_transfer: bool = False


class BaseAgent(metaclass=ABCMeta):
    """Root Neuroevolution agent class.

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
    """

    def __init__(self: "BaseAgent", config: BaseAgentConfig) -> None:
        """Initializes the agent seed."""
        self.config = config
        self.curr_seed = 0
        self.curr_run_score = 0
        self.curr_run_num_steps = 0
        if config.env_transfer:
            self.saved_env_state = None
            self.saved_env_out = None
            self.saved_env_seed = None
            self.curr_episode_score = 0
            self.curr_episode_num_steps = 0
        if config.fit_transfer:
            self.continual_fitness = 0

    @abstractmethod
    def mutate(self: "BaseAgent") -> None:
        """.

        Must be implemented.

        Args:
            seeds: An array of one or more random integers to seed the
                agent(s) mutation randomness.
        """
