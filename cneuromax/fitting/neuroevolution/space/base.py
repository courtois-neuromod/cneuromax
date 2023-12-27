""":class:`BaseSpace` and its config class."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Annotated as An

import numpy as np
from numpy.typing import NDArray

from cneuromax.fitting.neuroevolution.agent.singular import (
    BaseSingularAgent,
)
from cneuromax.utils.annotations import ge, le


@dataclass(frozen=True)
class BaseSpaceConfig:
    """Holds :class:`BaseSpace` config values.

    Args:
        eval_num_steps: Number of environment steps to run each agent\
            for during evaluation. `0` means that the agent will run\
            until the environment terminates (`eval_num_steps = 0` is\
            not supported for `env_transfer = True`).
        wandb_entity: See\
            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.wandb_entity`.
    """

    eval_num_steps: An[int, ge(0)]


@dataclass
class BaseSpaceAttrs:
    """Holds :class:`BaseSpace`"""


class BaseSpace(metaclass=ABCMeta):
    """

    Spaces are virtual environments in which agents produce behaviour
    and receive fitness scores.
    """

    def __init__(self: "BaseSpace", config: BaseSpaceConfig) -> None:
        """.

        Args:
            config: .
        """
        self.config = config

    @property
    def num_pops(self: "BaseSpace") -> An[int, ge(1), le(2)]:
        """Number of agents interacting in a given space.

        As of now, there are two optimization paradigms for spaces:
        - Reinforcement Spaces, that only utilize 1 population of
            actor/generator agents.
        - Imitation Spaces, that utilize 2 populations of agents:
            actor/generator agents and discriminator agents.

        For a given evaluation, a "regular" space makes interact one
        agent from each population, whereas a batch space makes
        interact N agents from each population in parallel. See
        :class:`~BaseSpace.evaluates_on_gpu`.
        """
        raise NotImplementedError

    @property
    def evaluates_on_gpu(self: "BaseSpace") -> bool:
        """Whether this space evaluates agents on GPU or not.

        As of now, there are two execution paradigms for spaces:
        - CPU execution, where agents are evaluated sequentially on CPU.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self: "BaseSpace",
        agent_s: list[list[BaseSingularAgent]],
        curr_gen: An[int, ge(1)],
    ) -> NDArray[np.float32]:
        """.

        Method called once per iteration (every generation) in order to
        evaluate and attribute fitnesses to agents.

        Args:
            agent_s: Agent(s) to evaluate.
            curr_gen: Current generation.

        Returns:
            The evaluation information: fitnesses and number of steps
                ran.
        """
