from abc import ABCMeta, abstractmethod
from typing import Annotated as An
from typing import final

import numpy as np
import wandb
from tensordict import TensorDict
from torch import Tensor
from torchrl.envs import EnvBase, GymEnv

from cneuromax.fitting.neuroevolution.agent.singular import BaseSingularAgent
from cneuromax.fitting.neuroevolution.space.base import BaseSpace
from cneuromax.utils.annotations import ge


class BaseImitationTarget(metaclass=ABCMeta):
    """Base Target class."""

    @abstractmethod
    def reset(self: "BaseImitationTarget", seed: int, step_num: int) -> None:
        """Reset the target's state given a seed and step number.

        Args:
            seed: Seed.
            step_num: Current step number.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self: "BaseImitationTarget", x: Tensor) -> Tensor:
        """Inputs a value and returns an output.
        Reset the target's state

        Args:
            x: Input value.
        Returns:
            The output value.
        """
        raise NotImplementedError


class BaseImitationSpace(BaseSpace, metaclass=ABCMeta):
    """Base Imitation Space class.

    Inside Imitation Spaces, agents evolve
    to imitate a target.
    """

    @property
    def imitation_target(self: "BaseImitationSpace") -> BaseImitationTarget:
        """The target to imitate."""
        raise NotImplementedError

    @property
    def hide_score(self: "BaseImitationSpace", obs: Tensor) -> Tensor:
        """Function that hides the environment's score portion of the screen
        to prevent the discriminator agent from utilizing it."""
        return obs

    @property
    def envs(self: "BaseImitationSpace") -> EnvBase | tuple[EnvBase, EnvBase]:
        """One or two environments to run the generator and target."""
        raise NotImplementedError

    @property
    @final
    def init_reset(
        self: "BaseImitationSpace", curr_gen: An[int, ge(0)]
    ) -> TensorDict:
        """
        First reset function called during the match.
        Used to either set the env seed or resume from a previous state.

        Args:
            curr_gen: Current generation.
        Returns:
            The initial environment observation.
        """
        if self.discriminator.config.env_transfer:
            if curr_gen == 0:
                self.curr_actor_data_holder.saved_env_seed = curr_gen

            obs = self.curr_env.reset(
                self.curr_actor_data_holder.saved_env_seed
            )

            if curr_gen > 0:
                self.curr_env.set_state(
                    self.curr_actor_data_holder.saved_env_state
                )

                obs = self.curr_actor_data_holder.saved_env_obs.copy()

            if self.imitation_target == self.curr_actor:
                self.imitation_target.reset(
                    self.curr_actor_data_holder.saved_env_seed,
                    self.curr_actor_data_holder.curr_episode_num_steps,
                )

        else:  # cfg.agent.gen_transfer in ["none", "fit"]:
            obs = self.curr_env.reset(curr_gen)

            if self.imitation_target == self.curr_actor:
                self.imitation_target.reset(curr_gen, 0)

        return obs

    @final
    def done_reset(self: "BaseImitationSpace", curr_gen: int) -> np.ndarray:
        """
        Reset function called whenever the env returns done.

        Args:
            curr_gen: Current generation.
        Returns:
            A new environment observation (np.ndarray).
        """
        if self.generator == self.curr_actor:
            self.generator.reset()
        self.discriminator.reset()
        if self.generator.config.env_transfer:
            if self.generator == self.curr_actor:
                if self.config.wandb_entity:
                    wandb.log(
                        {
                            "score": self.generator.curr_episode_score,
                            "gen": curr_gen,
                        },
                    )
                self.generator.curr_episode_score = 0
            self.curr_actor_data_holder.curr_episode_num_steps = 0

            self.curr_actor_data_holder.saved_env_seed = curr_gen

            obs = self.curr_env.reset(
                self.curr_actor_data_holder.saved_env_seed
            )

            return obs

        else:  # cfg.agent.gen_transfer in ["none", "fit"]:
            return np.empty(0)

    @final
    def final_reset(
        self: "BaseImitationSpace",
        obs: np.ndarray,
        curr_gen: An[int, ge(1)],
    ) -> None:
        """
        Reset function called at the end of every match.

        Args:
            obs: The final environment observation.
        """
        if not self.generator.config.mem_transfer:
            if self.generator == self.curr_actor:
                self.generator.reset()
            self.discriminator.reset()
        if self.generator.config.env_transfer:
            self.curr_actor_data_holder.saved_env_state = self.get_env_state(
                self.curr_env
            )

            self.curr_actor_data_holder.saved_env_obs = obs.copy()

        if cfg.agent.gen_transfer in ["none", "fit"]:
            if self.generator == self.curr_actor:  # check target
                if cfg.wandb != "disabled":
                    wandb.log(
                        {
                            "score": self.generator.curr_run_score,
                            "gen": curr_gen,
                        }
                    )

    @final
    def evaluate(
        self: "BaseImitationSpace",
        agent_s: list[list[BaseSingularAgent]],
        curr_gen: An[int, ge(1)],
    ) -> np.ndarray:
        self.generator = agent_s[0][0]
        self.discriminator = agent_s[0][1]
        generator_fitness, discriminator_fitness = 0, 0
        for match in [0, 1]:
            # Match 0: Generator & Discriminator
            # Match 1: Imitation Target & Discriminator
            self.curr_env = self.envs[-match]  # 1 or 2 envs
            if match == 0:
                # Match 0, actor is generator
                # Generator holds its own evaluation data
                self.curr_actor = self.generator
                self.curr_actor_data_holder = self.generator
            else:  # match == 1:
                # Match 1, actor is imitation target
                # Discriminator holds the target's evaluation data
                self.curr_actor = self.imitation_target
                self.curr_actor_data_holder = self.discriminator
            curr_run_score = 0
            curr_run_num_steps = 0
            obs, done, p_imitation_target = self.init_reset(curr_gen), False, 0
            hidden_score_obs = self.hide_score(obs, cfg.env)

            while not done:
                if self.generator == self.curr_actor:
                    output = self.generator(hidden_score_obs)
                else:  # self.imitation_target == self.curr_actor:
                    output = self.imitation_target(obs)

                obs, rew, done = self.curr_env.step(output)
                hidden_score_obs = self.hide_score(obs, cfg.env)

                curr_run_score += rew
                curr_run_num_steps += 1

                if self.discriminator.config.env_transfer:
                    self.curr_actor_data_holder.curr_episode_score += rew
                    self.curr_actor_data_holder.curr_episode_num_steps += 1

                p_imitation_target += self.discriminator(hidden_score_obs)

                if self.imitation_target == self.curr_actor:
                    if self.imitation_target.is_done:
                        done = True

                if done:
                    obs, done = self.done_reset(curr_gen)

                if (
                    self.curr_actor_data_holder.curr_run_num_steps
                    == cfg.agent.run_num_steps
                ):
                    done = True

            p_imitation_target /= (
                self.curr_actor_data_holder.curr_run_num_steps
            )

            if self.generator == self.curr_actor:
                generator_fitness += p_imitation_target
                discriminator_fitness -= p_imitation_target
            else:  # self.imitation_target == self.curr_actor:
                discriminator_fitness += p_imitation_target

            self.final_reset(obs)

        if cfg.agent.pop_merge:
            # Scale generator & discriminator fitnesses to [0, .5]
            generator_fitness = generator_fitness / 2
            discriminator_fitness = (discriminator_fitness + 1) / 4

        else:
            # Scale discriminator fitnesses to [0, 1]
            discriminator_fitness = (discriminator_fitness + 1) / 2

        if "fit" in cfg.agent.gen_transfer:
            self.generator.continual_fitness += generator_fitness
            self.discriminator.continual_fitness += discriminator_fitness

            return np.array(
                (
                    self.generator.continual_fitness,
                    self.discriminator.continual_fitness,
                ),
                (
                    self.generator.curr_run_num_steps,
                    self.discriminator.curr_run_num_steps,
                ),
            )

        else:
            return np.array(
                (
                    generator_fitness,
                    discriminator_fitness,
                ),
                (
                    self.generator.curr_run_num_steps,
                    self.discriminator.curr_run_num_steps,
                ),
            )
