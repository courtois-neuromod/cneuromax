""":mod:`cneuromax` entrypoint."""

from cneuromax.runner import BaseScheduleStarter, BaseTaskRunner
from cneuromax.utils.runner import get_runner_class

if __name__ == "__main__":
    Runner: type[BaseTaskRunner] = get_runner_class()
    TaskRunner.store_configs_and_start_runs()
