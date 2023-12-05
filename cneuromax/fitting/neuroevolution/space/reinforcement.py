"""."""

from abc import ABCMeta, abstractmethod
from typing import final

import numpy as np
import wandb

from cneuromax.fitting.neuroevolution.space.base import BaseSpace


class ReinforcementSpace(BaseSpace, metaclass=ABCMeta):
    """
    Base Reinforcement Space class. Inside Reinforcement Spaces, agents evolve
    to maximize a reward function.
    Concrete subclasses need to be named *Space*.
    """

    def __init__(self: "ReinforcementSpace") -> None:
        assert hasattr(self, "env")

        # cfg.agent.gen_transfer can be either bool or string
        assert not (
            cfg.agent.run_num_steps == "infinite"
            and "env" in cfg.agent.gen_transfer
        )

        super().__init__(io_path="gran.nevo.IO.base", num_pops=1)

    @property
    def num_pops(self: "BaseSpace") -> int:
        """See :class:`~.BaseSpace.num_pops`."""
        return 1

    @property
    def evaluates_on_gpu(self: "BaseSpace") -> bool:
        """See :class:`~.BaseSpace.evaluates_on_gpu`."""
        raise NotImplementedError

    @final
    def init_reset(self: "ReinforcementSpace", curr_gen: int) -> np.ndarray:
        """
        First reset function called during the run. Used to reset the
        environment & potentially resume from a previous state.

        Args:
            curr_gen: Current generation.
        Returns:
            np.ndarray: The initial environment observation.
        """
        if "env" in cfg.agent.gen_transfer:
            if curr_gen == 0:
                self.agent.saved_env_seed = curr_gen

            obs = self.env.reset(self.agent.saved_env_seed)

            if curr_gen > 0:
                self.env.set_state(self.agent.saved_env_state)

                obs = self.agent.saved_env_obs.copy()

        else:  # cfg.agent.gen_transfer in ["none", "fit"]:
            obs = self.env.reset(curr_gen)

        return obs

    @final
    def done_reset(self: "ReinforcementSpace", curr_gen: int) -> np.ndarray:
        """
        Reset function called whenever the environment returns done.

        Args:
            curr_gen: Current generation.
        Returns:
            np.ndarray: A new environment observation (np.ndarray).
        """

        if "env" in cfg.agent.gen_transfer:
            if cfg.wandb != "disabled":
                wandb.log(
                    {"score": self.agent.curr_episode_score, "gen": curr_gen}
                )

            self.agent.curr_episode_score = 0
            self.agent.curr_episode_num_steps = 0

            self.agent.saved_env_seed = curr_gen

            obs = self.env.reset(self.agent.saved_env_seed)

            return obs

        else:  # cfg.agent.gen_transfer in ["none", "fit"]:
            return np.empty(0)

    @final
    def final_reset(self: "ReinforcementSpace", obs: np.ndarray) -> None:
        """
        Reset function called at the end of every run.

        Args:
            obs: The final environment observation.
        """
        if "mem" not in cfg.agent.gen_transfer:
            self.agent.reset()

        if "env" in cfg.agent.gen_transfer:
            self.agent.saved_env_state = self.env.get_state()
            self.agent.saved_env_obs = obs.copy()

        if cfg.agent.gen_transfer in ["none", "fit"]:
            if cfg.wandb != "disabled":
                wandb.log(
                    {
                        "score": self.agent.curr_run_score,
                        "gen": curr_gen,
                    }
                )

    @final
    def run_agents(self: "ReinforcementSpace", curr_gen: int) -> float:
        [self.agent] = self.agents
        self.agent.curr_run_score = 0
        self.agent.curr_run_num_steps = 0

        obs, done = self.init_reset(curr_gen), False

        while not done:
            obs, rew, done = self.env.step(self.agent(obs))

            self.agent.curr_run_score += rew
            self.agent.curr_run_num_steps += 1

            if "env" in cfg.agent.gen_transfer:
                self.agent.curr_episode_score += rew
                self.agent.curr_episode_num_steps += 1

            if "fit" in cfg.agent.gen_transfer:
                self.agent.continual_fitness += rew

            if done:
                obs = self.done_reset(curr_gen)

            if self.agent.curr_run_num_steps == cfg.agent.run_num_steps:
                done = True

        self.final_reset(obs)

        if "fit" in cfg.agent.gen_transfer:
            return np.array(
                (self.agent.continual_fitness, self.agent.curr_run_num_steps)
            )

        else:
            return np.array(
                (self.agent.curr_run_score, self.agent.curr_run_num_steps)
            )
