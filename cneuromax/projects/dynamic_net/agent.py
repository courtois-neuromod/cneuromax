""":class:`TransferAgent` & :class:`TransferAgentConfig`."""

from dataclasses import dataclass
from typing import Annotated as An

import torch
import torch.nn.functional as f
from jaxtyping import Float, Int
from torch import Tensor
from torchrl.data.tensor_specs import ContinuousBox
from torchrl.envs.libs.gym import GymEnv

from cneuromax.fitting.neuroevolution.agent import BaseAgent, BaseAgentConfig
from cneuromax.fitting.neuroevolution.net.cpu.dynamic import (
    DynamicNet,
    DynamicNetConfig,
)
from cneuromax.utils.beartype import ge, le, one_of


@dataclass
class AgentConfig(BaseAgentConfig):  # noqa: D101
    env_name: str = "${space.config.env_name}"


class Agent(BaseAgent):  # noqa: D101

    def __init__(
        self: "Agent",
        config: AgentConfig,
        pop_idx: An[int, ge(0), le(1)],
        *,
        pops_are_merged: bool,
    ) -> None:
        super().__init__(
            config=config,
            pop_idx=pop_idx,
            pops_are_merged=pops_are_merged,
        )
        self.config: AgentConfig
        temp_env = GymEnv(env_name=config.env_name)
        self.num_actions = temp_env.action_spec.shape.numel()
        self.net = DynamicNet(
            config=DynamicNetConfig(
                num_inputs=temp_env.observation_spec[
                    "observation"
                ].shape.numel(),
                num_outputs=self.num_actions,
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

    def mutate(self: "Agent") -> None:
        """Mutates the agent."""
        self.net.mutate()

    def reset(self: "Agent") -> None:
        """Resets the agent's memory state."""
        self.net.reset()

    def __call__(
        self: "Agent",
        x: Float[Tensor, " obs_size"],
    ) -> Float[Tensor, " act_size"] | Int[Tensor, " act_size"]:
        """Forward pass.

        Args:
            x: The input observation.

        Returns:
            The output action.
        """
        x: list[float] = x.tolist()
        x: list[float] = self.net(x)
        x: Float[Tensor, " act_size"] = torch.tensor(x)
        x: Float[Tensor, " act_size"] | Int[Tensor, " act_size"] = (
            self.net_to_env(x=x)
        )
        return x

    def net_to_env(
        self: "Agent",
        x: Float[Tensor, " act_size"],
    ) -> Float[Tensor, " act_size"] | Int[Tensor, " act_size"]:
        """Processes the network output before feeding it to the env.

        Args:
            x: The network output.

        Returns:
            The network output processed for the env.
        """
        if self.output_mode == "discrete":
            x_d: Float[Tensor, " act_size"] = torch.softmax(input=x, dim=0)
            x_d: Int[Tensor, " "] = torch.multinomial(
                input=x_d,
                num_samples=1,
            ).squeeze()
            # Turn the integer into a one-hot vector.
            x_d: Int[Tensor, " act_size"] = f.one_hot(
                x_d,
                num_classes=self.num_actions,
            )
            return x_d
        else:  # self.output_mode == "continuous"  # noqa: RET505
            x_c: Float[Tensor, " act_size"] = torch.tanh(input=x)
            x_c: Float[Tensor, " act_size"] = (
                x_c * (self.output_high - self.output_low) / 2
                + (self.output_high + self.output_low) / 2
            )
            return x_c
