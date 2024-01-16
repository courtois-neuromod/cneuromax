""":class:`GymAgent` & :class:`GymAgentConfig`."""
from typing import Annotated as An

import torch
from jaxtyping import Float32
from torch import Tensor
from torchrl.envs.libs.gym import GymEnv

from cneuromax.fitting.neuroevolution.agent import BaseAgent, BaseAgentConfig
from cneuromax.fitting.neuroevolution.net.cpu.static import (
    CPUStaticRNNFC,
    CPUStaticRNNFCConfig,
)
from cneuromax.utils.beartype import ge, le, one_of
from cneuromax.utils.torch import RunningStandardization


class GymAgentConfig(BaseAgentConfig):
    """:class:`CPUStaticRNNFC` config values.

    Args:
        env_name: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.env_name`.
        hidden_size: Size of the RNN hidden state.
    """

    env_name: str = "${space.env_name}"


class GymAgent(BaseAgent):
    """Gym Feature-Based Control Static Agent.

    Args:
        config: See :paramref:`~BaseAgent.config`.
        pop_idx: See :paramref:`~BaseAgent.pop_idx`.
        pops_are_merged: See :paramref:`~BaseAgent.pops_are_merged`.
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
        temp_env = GymEnv(env_name=config.env_name)
        self.net = CPUStaticRNNFC(
            config=CPUStaticRNNFCConfig(
                input_size=temp_env.observation_spec["observation"].shape,
                hidden_size=50,
                output_size=temp_env.action_spec.shape,
            ),
        )
        self.output_mode: An[
            str,
            one_of("continuous", "discrete"),
        ] = temp_env.action_spec.domain
        if self.output_mode == "continuous":
            self.output_low = temp_env.action_spec.space.low
            self.output_high = temp_env.action_spec.space.high
        self.standardizer = RunningStandardization(self.net.rnn.input_size)

    def __call__(
        self: "GymAgent",
        x: Float32[Tensor, " obs_size"],
    ) -> Float32[Tensor, " #act_size"]:
        """Forward pass.

        Args:
            x: The input observation.

        Returns:
            The output action.
        """
        x: Float32[Tensor, " obs_size"] = self.env_to_net(x)
        x: Float32[Tensor, " act_size"] = self.net(x)
        x: Float32[Tensor, " #act_size"] = self.net_to_env(x)
        return x

    def env_to_net(
        self: "GymAgent",
        x: Float32[Tensor, " obs_size"],
    ) -> Float32[Tensor, " out_size"]:
        """Processes the observation before feeding it to the network.

        Args:
            x: The input observation.

        Returns:
            The processed observation.
        """
        x: Float32[Tensor, " obs_size"] = self.standardizer(x=x)
        return x

    def net_to_env(
        self: "GymAgent",
        x: Float32[Tensor, " act_size"],
    ) -> Float32[Tensor, " #act_size"]:
        """Processes the network output before feeding it to the env.

        Args:
            x: The network output.

        Returns:
            The processed network output.
        """
        if self.output_mode == "discrete":
            x_d: Float32[Tensor, " act_size"] = torch.softmax(input=x, dim=0)
            x_d: Float32[Tensor, " 1"] = torch.multinomial(
                input=x_d,
                num_samples=1,
            )
            return x_d
        else:  # self.output_mode == "continuous"  # noqa: RET505
            x_c: Float32[Tensor, " act_size"] = torch.tanh(input=x)
            x_c: Float32[Tensor, " act_size"] = (
                x_c * (self.output_high - self.output_low) / 2
                + (self.output_high + self.output_low) / 2
            )
            return x_c
