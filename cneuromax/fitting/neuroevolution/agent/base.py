"""."""

from abc import ABCMeta, abstractmethod


class BaseAgent(metaclass=ABCMeta):
    """Root Neuroevolution agent class."""

    def __init__(self: "BaseAgent") -> None:
        """Initializes the agent seed."""
        self.curr_seed = 0

    @abstractmethod
    def mutate(self: "BaseAgent") -> None:
        """.

        Must be implemented.

        Args:
            seeds: An array of one or more random integers to seed the
                agent(s) mutation randomness.
        """
