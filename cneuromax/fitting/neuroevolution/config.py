""":mod:`hydra-core` Neuroevolution fitting config & validation."""

from dataclasses import dataclass
from typing import Annotated as An
from typing import Any

from omegaconf import MISSING, DictConfig

from cneuromax.fitting.config import (
    BaseFittingHydraConfig,
    post_process_base_fitting_config,
    pre_process_base_fitting_config,
)
from cneuromax.utils.annotations import ge


@dataclass
class NeuroevolutionFittingHydraConfig(BaseFittingHydraConfig):
    """.

    Args:
        space: Implicit (generated by :mod:`hydra-zen`)\
            `SpaceHydraConfig` instance.
        agent: Implicit (generated by :mod:`hydra-zen`)\
            `AgentHydraConfig` instance.
        wandb_entity: :mod:`wandb` entity (username or team name)\
            to use for logging. `None` means no logging.
        agents_per_task: Number of agents per task (`num_tasks` =\
            `num_nodes` x `tasks_per_node`).
        prev_num_gens: Number of generations from a previous experiment\
            to load.
        total_num_gens: Number of generations to run the experiment for\
            (including the previous number of generations).
        save_interval: Number of generations between each save point.\
            `0` means no save point except for the last generation.
        save_first_gen: Whether to save the state of the experiment\
            after the first generation (usually for plotting purposes).
        pop_merge: Whether to merge both generator and discriminator\
            populations into a single population. This means that each\
            agent will be evaluated on both its generative and\
            discriminative abilities.
        env_transfer: Whether a parent agent's environment state\
            (position, velocity, ...) is transferred to its children.
        fit_transfer: Whether a parent agent's fitness is transferred\
            to its children.
        mem_transfer: Whether a parent agent's hidden state is\
            transferred to its children.
        eval_num_steps: Number of environment steps to run each agent\
            for during evaluation. `0` means that the agent will run\
            until the environment terminates (`eval_num_steps = 0` is\
            not supported for `env_transfer = True`).
    """

    space: Any = MISSING
    agent: Any = MISSING
    wandb_entity: str | None = None
    agents_per_task: An[int, ge(1)] = 1
    prev_num_gens: An[int, ge(0)] = 0
    total_num_gens: An[int, ge(1)] = 10
    save_interval: An[int, ge(0)] = 0
    save_first_gen: bool = False
    pop_merge: bool = False
    env_transfer: bool = False
    fit_transfer: bool = False
    mem_transfer: bool = False
    eval_num_steps: An[int, ge(0)] = 0


def pre_process_neuroevolution_fitting_config(config: DictConfig) -> None:
    """Pre-processes config from :func:`hydra.main` before resolution.

    Args:
        config: The not yet processed :mod:`hydra-core` config.
    """
    pre_process_base_fitting_config(config)
    if config.eval_num_steps == 0 and config.env_transfer:
        error_msg = "`env_transfer = True` requires `eval_num_steps > 0`."
        raise ValueError(error_msg)
    if (
        config.total_num_gens - config.prev_num_gens
    ) % config.save_interval != 0:
        error_msg = (
            "`save_interval` must be a multiple of "
            "`total_num_gens - prev_num_gens`."
        )
        raise ValueError(error_msg)


def post_process_neuroevolution_fitting_config(
    config: NeuroevolutionFittingHydraConfig,
) -> None:
    """Post-processes the :mod:`hydra-core` config after it is resolved.

    Args:
        config: The processed :mod:`hydra-core` config.
    """
    post_process_base_fitting_config(config)
