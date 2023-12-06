"""."""

from cneuromax.fitting.neuroevolution.agent.base import BaseAgent


class BaseSingularAgent(BaseAgent):
    def reset(self) -> None:
        """Reset the agent's state."""
        raise NotImplementedError
