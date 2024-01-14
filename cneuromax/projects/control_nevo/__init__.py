"""Control task neuroevolution ``project``.

``__main__.py``:

.. highlight:: python
.. code-block:: python

    from cneuromax.projects.control_nevo import TaskRunner

    if __name__ == "__main__":
        TaskRunner.store_configs_and_run_task()
"""
from hydra_zen import ZenStore, make_config

from cneuromax.fitting.neuroevolution.runner import NeuroevolutionTaskRunner
from cneuromax.utils.hydra_zen import fs_builds

__all__ = [
    "TaskRunner",
]


class TaskRunner(NeuroevolutionTaskRunner):
    """MNIST classification ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra-core` MNIST classification configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store)
        task_store = store(group="task", package="_global_")
        task_store(
            make_config(
                hydra_defaults=["_self_", {"override /db": "sqlite"}],
                server=dict(port=8080),
                bases=(Config,),
            ),
            name="aplite",
        )
