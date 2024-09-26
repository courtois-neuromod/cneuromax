"""Neuroevolution ``subtask`` (regular + test) and ``task`` configs."""

from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import wandb
from hydra_zen import make_config

from cneuromax.fitting.config import FittingSubtaskConfig
from cneuromax.fitting.neuroevolution.agent import BaseAgent
from cneuromax.fitting.neuroevolution.space import BaseSpace
from cneuromax.utils.beartype import ge
from cneuromax.utils.hydra_zen import builds, fs_builds, p_builds, pfs_builds


@dataclass
class NeuroevolutionSubtaskConfig(FittingSubtaskConfig):
    """Neuroevolution ``subtask`` config.

    Args:
        agents_per_task: Number of agents per task (``num_tasks`` =
            ``num_nodes`` x ``tasks_per_node``).
        total_num_gens: Number of generations to run the experiment for
            (including the previous number of generations).
        save_interval: Number of generations between each save point.
            `0` means no save point except for the last generation.
        save_first_gen: Whether to save the state of the experiment
            after the first generation (usually for plotting purposes).
        pop_merge: Whether to merge both generator and discriminator
            populations into a single population. This means that each
            agent will be evaluated on both its generative and
            discriminative abilities.
        env_transfer: Whether an agent's environment state
            (position, velocity, ...) is transferred to its children
            if it passes through the selection process.
        fit_transfer: Whether an agent's fitness is transferred to
            its children if it passes through the selection process.
        mem_transfer: Whether an agent's memory (hidden state) is
            transferred to its children if it passes through the
            selection process.
        eval_num_steps: Number of environment steps to run each agent
            for during evaluation. ``0`` means that the agent will run
            until the environment terminates (``eval_num_steps = 0`` is
            not supported for ``env_transfer = True``).
        logging: Whether to log the experiment to Weights & Biases.
    """

    agents_per_task: An[int, ge(1)] = 1
    total_num_gens: An[int, ge(1)] = 10
    save_interval: An[int, ge(0)] = 0
    save_first_gen: bool = False
    pop_merge: bool = False
    env_transfer: bool = False
    fit_transfer: bool = False
    mem_transfer: bool = False
    eval_num_steps: An[int, ge(0)] = 0
    logging: bool = True


@dataclass
class NeuroevolutionSubtaskTestConfig(NeuroevolutionSubtaskConfig):
    """Neuroevolution ``subtask`` test config.

    Args:
        num_tests: Number of episodes to evaluate each agent on.
        test_num_steps: Number of environment steps to run each agent
            for during testing. ``0`` means that the agent will run
            until the environment terminates.
        logging: See :paramref:`.NeuroevolutionSubtaskConfig.logging`.
    """

    num_tests: An[int, ge(1)] = 2
    test_num_steps: An[int, ge(0)] = 0
    logging: bool = False


@dataclass
class NeuroevolutionTaskConfig(
    make_config(  # type: ignore[misc]
        space=builds(BaseSpace),
        agent=p_builds(BaseAgent),
        logger=pfs_builds(wandb.init),
        config=fs_builds(NeuroevolutionSubtaskConfig),
    ),
):
    """Neuroevolution ``task`` config.

    Args:
        defaults: Hydra defaults.
        space: See :class:`.BaseSpace`.
        agent: See :class:`.BaseAgent`.
        logger: See :func:`wandb.init`.
        config: See :class:`.NeuroevolutionSubtaskConfig`.
    """

    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"logger": "wandb_simexp"},
            "project",
            "task",
            {"task": None},
        ],
    )
