""":class:`GymAgent` & its config."""

import random
from dataclasses import dataclass
from typing import Annotated as An

import torch
from jaxtyping import Float32, Int64
from torch import Tensor
from torchrl.data.tensor_specs import ContinuousBox
from torchrl.envs.libs.gym import GymEnv

from cneuromax.fitting.neuroevolution.agent import BaseAgent, BaseAgentConfig
from cneuromax.fitting.neuroevolution.net.cpu.dynamic import (
    DynamicNet,
    DynamicNetConfig,
)
from cneuromax.projects.neuroevorl_control.agent import GymAgent
from cneuromax.utils.beartype import ge, le, one_of
from cneuromax.utils.torch import RunningStandardization


@dataclass
class DynamicGymAgentConfig(BaseAgentConfig):
    """Holds :class:`DynamicGymAgent` config values.

    Args:
        env_name: See\
            :paramref:`~.GymReinforcementSpaceConfig.env_name`.
    """

    env_name: str = "${space.config.env_name}"


class DynamicGymAgent(GymAgent):
    """``project`` :class:`DynamicGymAgent`.

    Args:
        config: See :paramref:`~.BaseAgent.config`.
        pop_idx: See :paramref:`~.BaseAgent.pop_idx`.
        pops_are_merged: See :paramref:`~.BaseAgent.pops_are_merged`.
    """

    def __init__(
        self: "GymAgent",
        config: DynamicGymAgentConfig,
        pop_idx: An[int, ge(0), le(1)],
        *,
        pops_are_merged: bool,
    ) -> None:
        BaseAgent.__init__(
            self=self,
            config=config,
            pop_idx=pop_idx,
            pops_are_merged=pops_are_merged,
        )
        self.config: DynamicGymAgentConfig
        temp_env = GymEnv(env_name=config.env_name)
        self.num_actions = temp_env.action_spec.shape.numel()
        self.net: DynamicNet = DynamicNet(  # type: ignore[assignment]
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
        self.standardizer = RunningStandardization(self.net.config.num_inputs)

    def mutate(self: "GymAgent") -> None:
        """Mutates the agent."""
        rand_bool = bool(random.getrandbits(1))
        if rand_bool:
            self.net.grow_node()

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
        x: Float32[Tensor, " act_size"] = torch.tensor(self.net(x=x))
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
