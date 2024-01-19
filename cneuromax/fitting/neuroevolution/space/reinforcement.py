""":class:`BaseReinforcementSpace`."""
import copy
from abc import ABCMeta
from typing import Annotated as An
from typing import Any, final

import numpy as np
from tensordict import TensorDict
from torchrl.envs import EnvBase

from cneuromax.fitting.neuroevolution.agent import BaseAgent
from cneuromax.fitting.neuroevolution.space.base import (
    BaseSpace,
    BaseSpaceConfig,
)
from cneuromax.fitting.neuroevolution.utils.wandb import gather
from cneuromax.utils.beartype import ge


class BaseReinforcementSpace(BaseSpace, metaclass=ABCMeta):
    """Reinforcement Base Space class.

    Args:
        env: The :mod:`torchrl` environment to run the evaluation on.
        config: See :paramref:`~.BaseSpace.config`.
    """

    def __init__(
        self: "BaseReinforcementSpace",
        config: BaseSpaceConfig,
        env: EnvBase,
    ) -> None:
        super().__init__(config=config, num_pops=1, evaluates_on_gpu=False)
        self.env = env

    @final
    def run_pre_eval(
        self: "BaseReinforcementSpace",
        agent: BaseAgent,
        curr_gen: int,
    ) -> TensorDict:
        """Resets/loads the environment before evaluation begins.

        Args:
            agent: The agent being evaluated.
            curr_gen: See :paramref:`~.BaseSpace.curr_gen`.

        Returns:
            See :paramref:`run_post_eval.out`.
        """
        if curr_gen > 1 and agent.config.env_transfer:
            self.env = copy.deepcopy(agent.saved_env)
            return copy.deepcopy(agent.saved_env_out)
        self.env.set_seed(seed=curr_gen)
        return self.env.reset()

    @final
    def env_done_reset(
        self: "BaseReinforcementSpace",
        agent: BaseAgent,
        out: TensorDict,
        curr_gen: int,
    ) -> TensorDict | dict[str, bool]:
        """Resets the agent/environment when the environment terminates.

        Args:
            agent: See :paramref:`pre_eval_reset.agent`.
            out: The latest environment output.
            curr_gen: See :paramref:`~.BaseSpace.curr_gen`.

        Returns:
            See :paramref:`run_post_eval.out`.
        """
        # env,fit,env+fit,env+fit+mem: reset, mem,mem+fit: no reset
        if not (
            agent.config.mem_transfer
            or (agent.config.mem_transfer and agent.config.fit_transfer)
        ):
            agent.reset()
        if agent.config.env_transfer:
            self.logged_score: float | None = agent.curr_episode_score
            agent.curr_episode_score = 0
            agent.curr_episode_num_steps = 0
            self.env.set_seed(seed=curr_gen)
            return self.env.reset()
        return out

    @final
    def run_post_eval(
        self: "BaseReinforcementSpace",
        agent: BaseAgent,
        out: TensorDict,
        curr_gen: int,
    ) -> None:
        """Resets the agent & saves the environment post-evaluation.

        Args:
            agent: See :paramref:`pre_eval_reset.agent`.
            out: The latest environment output.
            curr_gen: See :paramref:`~.BaseSpace.curr_gen`.
        """
        if not agent.config.mem_transfer:
            agent.reset()
        if agent.config.env_transfer:
            agent.saved_env = copy.deepcopy(self.env)
            agent.saved_env_out = copy.deepcopy(out)
        if not agent.config.env_transfer:
            self.logged_score = agent.curr_eval_score
        gather(logged_score=self.logged_score, curr_gen=curr_gen)

    @final
    def evaluate(
        self: "BaseReinforcementSpace",
        agents: list[list[BaseAgent]],
        curr_gen: An[int, ge(1)],
    ) -> np.ndarray[np.float32, Any]:
        """Evaluation function called once per generation.

        Args:
            agents: A 2D list containing the agent to evaluate.
            curr_gen: See :paramref:`~.BaseSpace.curr_gen`.
        """
        agent = agents[0][0]
        agent.curr_eval_score = 0
        agent.curr_eval_num_steps = 0
        self.logged_score = None
        out = self.run_pre_eval(agent=agent, curr_gen=curr_gen)
        while not out["done"]:
            out = out.set(key="action", item=agent(x=out["observation"]))
            out = self.env.step(tensordict=out)["next"]
            agent.curr_eval_score += float(out["reward"])
            agent.curr_eval_num_steps += 1
            if agent.config.env_transfer:
                agent.curr_episode_score += float(out["reward"])
                agent.curr_episode_num_steps += 1
            if agent.config.fit_transfer:
                agent.continual_fitness += float(out["reward"])
            if out["done"]:
                out = self.env_done_reset(
                    agent=agent,
                    out=out,
                    curr_gen=curr_gen,
                )
            if agent.curr_eval_num_steps == self.config.eval_num_steps:
                out["done"] = True
        self.run_post_eval(agent=agent, out=out, curr_gen=curr_gen)
        return np.array(
            (
                agent.continual_fitness
                if agent.config.fit_transfer
                else agent.curr_eval_score,
                agent.curr_eval_num_steps,
            ),
        )
