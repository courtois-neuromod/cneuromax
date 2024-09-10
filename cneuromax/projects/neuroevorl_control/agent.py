""":class:`GymAgent` & its config."""

from dataclasses import dataclass
from typing import Annotated as An

import torch
import torch.nn.functional as f
from jaxtyping import Float32, Int64
from torch import Tensor
from torchrl.data.tensor_specs import ContinuousBox
from torchrl.envs.libs.gym import GymEnv

from cneuromax.fitting.neuroevolution.agent import BaseAgent, BaseAgentConfig
from cneuromax.fitting.neuroevolution.net.cpu.static import (
    CPUStaticRNNFC,
    CPUStaticRNNFCConfig,
)
from cneuromax.utils.beartype import ge, le, one_of
from cneuromax.utils.torch import RunningStandardization


@dataclass
class GymAgentConfig(BaseAgentConfig):
    """Holds :class:`GymAgent` config values.

    Args:
        env_name: See
            :paramref:`~.GymReinforcementSpaceConfig.env_name`.
        hidden_size: Size of the RNN hidden state.
        mutation_std: Standard deviation of the mutation noise.
    """

    env_name: str = "${space.config.env_name}"
    hidden_size: int = 50
    mutation_std: float = 0.01


class GymAgent(BaseAgent):
    """``project`` :class:`BaseAgent`.

    Args:
        config: See :paramref:`~.BaseAgent.config`.
        pop_idx: See :paramref:`~.BaseAgent.pop_idx`.
        pops_are_merged: See :paramref:`~.BaseAgent.pops_are_merged`.
    """

    def __init__(
        self: "GymAgent",
        config: GymAgentConfig,
        pop_idx: An[int, ge(0), le(1)],
        *,
        pops_are_merged: bool,
    ) -> None:
        super().__init__(
            config=config,
            pop_idx=pop_idx,
            pops_are_merged=pops_are_merged,
        )
        self.config: GymAgentConfig
        temp_env = GymEnv(env_name=config.env_name)
        self.num_actions = temp_env.action_spec.shape.numel()
        self.net = CPUStaticRNNFC(
            config=CPUStaticRNNFCConfig(
                input_size=temp_env.observation_spec[
                    "observation"
                ].shape.numel(),
                hidden_size=config.hidden_size,
                output_size=self.num_actions,
            ),
        )
        self.output_mode: An[
            str,
            one_of("continuous", "discrete"),
        ] = temp_env.action_spec.domain
        if self.output_mode == "continuous":
            action_space: ContinuousBox = temp_env.action_spec.space
            self.output_low = action_space.low
            self.output_high = action_space.high
        self.standardizer = RunningStandardization(self.net.rnn.input_size)

    def mutate(self: "GymAgent") -> None:
        """Mutates the agent."""
        for param in self.net.parameters():
            param.data += self.config.mutation_std * torch.randn_like(
                input=param.data,
            )

    def reset(self: "GymAgent") -> None:
        """Resets the agent's memory state."""
        self.net.reset()

    def __call__(
        self: "GymAgent",
        x: Float32[Tensor, " obs_size"],
    ) -> Float32[Tensor, " act_size"] | Int64[Tensor, " act_size"]:
        """Forward pass.

        Args:
            x: The input observation.

        Returns:
            The output action.
        """
        x: Float32[Tensor, " obs_size"] = self.env_to_net(x=x)
        x: Float32[Tensor, " act_size"] = self.net(x=x)
        x: (
            Float32[
                Tensor,
                " act_size",
            ]
            | Int64[
                Tensor,
                " act_size",
            ]
        ) = self.net_to_env(x=x)
        return x

    def env_to_net(
        self: "GymAgent",
        x: Float32[Tensor, " obs_size"],
    ) -> Float32[Tensor, " out_size"]:
        """Processes the observation before feeding it to the network.

        Args:
            x: See :paramref:`~__call__.x`.

        Returns:
            The observation processed for the network.
        """
        x: Float32[Tensor, " obs_size"] = self.standardizer(x=x)
        return x

    def net_to_env(
        self: "GymAgent",
        x: Float32[Tensor, " act_size"],
    ) -> Float32[Tensor, " act_size"] | Int64[Tensor, " act_size"]:
        """Processes the network output before feeding it to the env.

        Args:
            x: The network output.

        Returns:
            The network output processed for the env.
        """
        if self.output_mode == "discrete":
            x_d: Float32[Tensor, " act_size"] = torch.softmax(input=x, dim=0)
            x_d: Int64[Tensor, " "] = torch.multinomial(
                input=x_d,
                num_samples=1,
            ).squeeze()
            # Turn the integer into a one-hot vector.
            x_d: Int64[Tensor, " act_size"] = f.one_hot(
                x_d,
                num_classes=self.num_actions,
            )
            return x_d
        else:  # self.output_mode == "continuous"  # noqa: RET505
            x_c: Float32[Tensor, " act_size"] = torch.tanh(input=x)
            x_c: Float32[Tensor, " act_size"] = (
                x_c * (self.output_high - self.output_low) / 2
                + (self.output_high + self.output_low) / 2
            )
            return x_c
