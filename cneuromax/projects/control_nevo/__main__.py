"""Control Tasks Neuroevolution ``project`` entrypoint."""
from cneuromax.projects.control_nevo import TaskRunner

if __name__ == "__main__":
    TaskRunner.store_configs_and_run_task()
