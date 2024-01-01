"""."""

from cneuromax.fitting.neuroevolution.agent.base import BaseAgent


class BaseSingularAgent(BaseAgent):
    def reset(self) -> None:
        """Reset the agent's state."""
        raise NotImplementedError

    @property
    def seed(self: "BaseSingularAgent") -> int:
        return self._seed

    @seed.setter
    def seed(self: "BaseSingularAgent", seed: int) -> None:
        self._seed = seed
