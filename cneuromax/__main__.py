""":mod:`cneuromax` entrypoint."""

from cneuromax.runner import BaseTaskRunner
from cneuromax.utils.runner import get_task_runner_class
from cneuromax.utils.wandb import login_wandb

if __name__ == "__main__":
    TaskRunner: type[BaseTaskRunner] = get_task_runner_class()
    login_wandb()
    TaskRunner.store_configs_and_run_task()
