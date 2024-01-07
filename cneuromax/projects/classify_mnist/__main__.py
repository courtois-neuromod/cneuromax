"""MNIST classification ``project`` entrypoint."""
from cneuromax.projects.classify_mnist import TaskRunner

if __name__ == "__main__":
    TaskRunner.store_configs_and_run_task()
