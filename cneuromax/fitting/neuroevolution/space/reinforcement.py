"""."""
from abc import ABCMeta
from typing import Annotated as An
from typing import final

import numpy as np
import wandb
from tensordict import TensorDict
from torchrl.envs import EnvBase

from cneuromax.fitting.neuroevolution.agent.singular import BaseSingularAgent
from cneuromax.fitting.neuroevolution.space.base import BaseSpace
from cneuromax.utils.annotations import ge


class BaseReinforcementSpace(BaseSpace, metaclass=ABCMeta):
    """Base Reinforcement Space class.

    Inside Reinforcement Spaces, agents evolve to maximize a reward
    function.
    """

    @property
    def env(self: "BaseReinforcementSpace") -> EnvBase:
        """Environment to run the agent."""
        raise NotImplementedError

    @property
    def num_pops(self: "BaseReinforcementSpace") -> int:
        """See :class:`~.BaseSpace.num_pops`."""
        return 1

    @property
    def evaluates_on_gpu(self: "BaseReinforcementSpace") -> bool:
        """See :class:`~.BaseSpace.evaluates_on_gpu`."""
        raise NotImplementedError

    @final
    def init_reset(
        self: "BaseReinforcementSpace", curr_gen: int
    ) -> TensorDict:
        """First reset function called during the execution.

        Used to reset the
        environment & potentially resume from a previous state.

        Args:
            curr_gen: See :paramref:`~.BaseSpace.curr_gen`.

        Returns:
            The initial environment observation.
        """
        if self.agent.config.env_transfer:
            if curr_gen == 1:
                self.agent.saved_env_seed = curr_gen
            self.env.set_seed(seeds=self.agent.saved_env_seed)
            out = self.env.reset()
            if curr_gen > 1:
                self.env.set_state(self.agent.saved_env_state)
                out = self.agent.saved_env_out.copy()
        else:
            self.env.set_seed(seeds=curr_gen)
            out = self.env.reset(curr_gen)
        return out

    @final
    def done_reset(
        self: "BaseReinforcementSpace",
        curr_gen: int,
    ) -> TensorDict:
        """Reset function called whenever the environment returns done.

        Args:
            curr_gen: See :paramref:`~.BaseSpace.curr_gen`.

        Returns:
            A new environment observation (np.ndarray).
        """
        if self.agent.config.env_transfer:
            if self.config.wandb_entity:
                wandb.log(
                    {"score": self.agent.curr_episode_score, "gen": curr_gen},
                )
            self.agent.curr_episode_score = 0
            self.agent.curr_episode_num_steps = 0
            self.agent.saved_env_seed = curr_gen
            self.env.set_seed(seeds=self.agent.saved_env_seed)
            out = self.env.reset(self.agent.saved_env_seed)
        else:
            out = TensorDict()
        return out

    @final
    def final_reset(
        self: "BaseReinforcementSpace",
        out: TensorDict,
        curr_gen: int,
    ) -> None:
        """Reset function called at the end of every run.

        Args:
            obs: The final environment observation.
        """
        if self.agent.config.mem_transfer:
            self.agent.reset()
        if self.agent.config.env_transfer:
            self.agent.saved_env_state = self.env.get_state()
            self.agent.saved_env_out = out.copy()
        if (
            not (
                self.agent.config.env_transfer
                or self.agent.config.mem_transfer
            )
            and self.config.wandb_entity
        ):
            wandb.log(
                {
                    "score": self.agent.curr_run_score,
                    "gen": curr_gen,
                },
            )

    @final
    def evaluate(
        self: "BaseReinforcementSpace",
        agent_s: list[list[BaseSingularAgent]],
        curr_gen: An[int, ge(1)],
    ) -> float:
        """."""
        self.agent = agent_s[0][0]
        self.agent.curr_run_score = 0
        self.agent.curr_run_num_steps = 0
        out = self.init_reset(curr_gen), False
        while not out["done"]:
            out = out.set("action", self.agent(out["obs"]))
            out = self.env.step(out)["next"]
            self.agent.curr_run_score += out["rew"]
            self.agent.curr_run_num_steps += 1
            if self.agent.config.env_transfer:
                self.agent.curr_episode_score += out["rew"]
                self.agent.curr_episode_num_steps += 1
            if self.agent.config.fit_transfer:
                self.agent.continual_fitness += out["rew"]
            if out["done"]:
                obs = self.done_reset(curr_gen)
            if self.agent.curr_run_num_steps == self.config.eval_num_steps:
                out["done"] = True
        self.final_reset(obs)
        return np.array(
            object=(
                self.agent.continual_fitness
                if self.agent.config.fit_transfer
                else self.agent.curr_run_score,
                self.agent.curr_run_num_steps,
            ),
        )
