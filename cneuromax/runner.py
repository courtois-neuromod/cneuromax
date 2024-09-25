""":class:`BaseTaskRunner`."""

from abc import ABC, abstractmethod
from typing import Any, final

from hydra_zen import ZenStore, zen

from cneuromax.config import BaseHydraConfig
from cneuromax.utils.hydra_zen import destructure
from cneuromax.utils.runner import (
    get_absolute_project_path,
    get_project_and_task_names,
)


class BaseTaskRunner(ABC):
    """``task`` runner.

    Stores configs and runs the ``task``.

    Attributes:
        hydra_config: The structured :class:`hydra.HydraConf` config
            used during the ``task`` execution.
    """

    hydra_config = BaseHydraConfig

    @final
    @classmethod
    def store_configs_and_run_task(cls: type["BaseTaskRunner"]) -> None:
        """Stores various configs and runs the ``task``.

        Args:
            cls: The :class:`BaseTaskRunner` subclass calling this
                method.
        """
        store = ZenStore()
        store(cls.hydra_config, name="config", group="hydra")
        project_name, task_name = get_project_and_task_names()
        store({"project": project_name}, name="project")
        store({"task": task_name}, name="task")
        # Hydra runtime type checking issues with structured configs:
        # https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/621#discussioncomment-7938326
        # `destructure` disables Hydra's runtime type checking, which is
        # fine since we use Beartype throughout the codebase.
        store = store(to_config=destructure)
        cls.store_configs(store=store)
        store.add_to_hydra_store(overwrite_ok=True)
        zen(cls.run_subtask).hydra_main(
            config_path=get_absolute_project_path(),
            config_name="config",
            version_base=None,
        )

    @classmethod
    @abstractmethod
    def store_configs(cls: type["BaseTaskRunner"], store: ZenStore) -> None:
        """Stores structured configs.

        Args:
            cls: See :paramref:`~store_configs_and_run_task.cls`.
            store: A :class:`hydra_zen.ZenStore` instance that manages
                the `hydra <https://hydra.cc>`_ configuration store.
        """

    @staticmethod
    @abstractmethod
    def run_subtask(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Runs the ``subtask`` given :paramref:`config`.

        This method is meant to hold the ``subtask`` execution logic.
        """
