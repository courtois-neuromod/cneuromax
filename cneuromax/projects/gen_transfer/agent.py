""":class:`TransferAgent` & :class:`TransferAgentConfig`."""
from dataclasses import dataclass
from typing import Annotated as An

from jaxtyping import Float, Float32
from torch import Tensor

from cneuromax.projects.neuroevorl_control import GymAgent, GymAgentConfig
from cneuromax.utils.beartype import one_of


@dataclass
class TransferAgentConfig(GymAgentConfig):
    """Holds :class:`TransferAgent` config values.

    Args:
        partial_obs: Whether the agent has partial observability.
    """

    partial_obs: bool = True
    env_name: An[
        str,
        one_of("Acrobot-v1", "HalfCheetah-v4"),
    ] = "${space.config.env_name}"


class TransferAgent(GymAgent):
    """``gen_transfer project`` :class:`BaseAgent`."""

    def env_to_net(
        self: "TransferAgent",
        x: Float[Tensor, " obs_size"],
    ) -> Float32[Tensor, " out_size"]:
        """Hides velocity from observation and calls parent method.

        Args:
            x: See :paramref:`~.GymAgent.env_to_net.x`.

        Returns:
            See return value of :meth:`~.GymAgent.env_to_net`.
        """
        self.config: TransferAgentConfig
        if self.config.partial_obs:
            hide_velocity(x=x, env_name=self.config.env_name)
        return super().env_to_net(x=x)


def hide_velocity(
    x: Float[Tensor, " obs_size"],
    env_name: An[str, one_of("Acrobot-v1", "HalfCheetah-v4")],
) -> None:
    """Hide velocity from observation.

    Args:
        x: See :paramref:`~.TransferAgent.env_to_net.x`.
        env_name: The name of the :mod:`gymnasium` environment.
    """
    # https://gymnasium.farama.org/environments/classic_control/acrobot/
    if env_name == "Acrobot-v1":
        x[4:] = 0
    # https://gymnasium.farama.org/environments/mujoco/half_cheetah/
    else:  # env_name == "HalfCheetah-v4":
        x[8:] = 0
