""":class:`TransferAgent` & :class:`TransferAgentConfig`."""
from typing import Annotated as An

from jaxtyping import Float32
from torch import Tensor

from cneuromax.projects.neuroevorl_control import GymAgent
from cneuromax.utils.beartype import one_of


class TransferAgent(GymAgent):
    """Transfer Feature-Based Control Static Agent."""

    def env_to_net(
        self: "TransferAgent",
        x: Float32[Tensor, " obs_size"],
    ) -> Float32[Tensor, " out_size"]:
        """Processes the observation before feeding it to the network.

        Hides the velocity from the observation and standardizes it.

        Args:
            x: The input observation.

        Returns:
            The processed observation.
        """
        hide_velocity(x=x, env_name=self.config.env_name)
        return super().env_to_net(x=x)


def hide_velocity(
    x: Float32[Tensor, " obs_size"],
    env_name: An[
        str,
        one_of(
            "Acrobot-v1",
            "CartPole-v1",
            "MountainCarContinuous-v0",
            "MountainCar-v0",
            "Pendulum-v1",
        ),
    ],
) -> None:
    """Hide velocity from observation.

    Args:
        x: See :paramref:`~.TransferAgent.env_to_net.x`.
        env_name: The name of the :mod:`gymnasium` environment.
    """
    # https://gymnasium.farama.org/environments/classic_control/acrobot/
    if env_name == "Acrobot-v1":
        x[4] = 0
        x[5] = 0
    # https://gymnasium.farama.org/environments/classic_control/cart_pole/
    elif env_name == "CartPole-v1":
        x[1] = 0
        x[3] = 0
    # https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/
    # https://gymnasium.farama.org/environments/classic_control/mountain_car/
    elif env_name in ["MountainCarContinuous-v0", "MountainCar-v0"]:
        x[1] = 0
    # https://gymnasium.farama.org/environments/classic_control/pendulum/
    else:  # env_name == "Pendulum-v1":
        x[2] = 0
