""":class:`BaseTaskRunner`.

Check-out :mod:`cneuromax` for an overview of the terms ``subtask``,
``task``, ``project``, ``service`` and ``interface``.
"""
import sys
from abc import ABC, abstractmethod
from typing import Any, ClassVar, final

from hydra.experimental.callbacks import LogJobReturnCallback
from hydra_zen import ZenStore, zen

from cneuromax.config import BaseHydraConfig, BaseSubtaskConfig
from cneuromax.utils.zen import fs_builds


class BaseTaskRunner(ABC):
    """``task`` runner.

    Stores configs and runs the ``task``.

    Attr:
        task_config_name: Name of the ``task`` config file. Must be\
            utilized in the ``service`` :meth:`store_configs` method.
        task_config_path: Path to the ``task`` config file, is set to\
            the ``project`` root directory.
        task_hydra_defaults: ``task`` :mod:`hydra` defaults. Must be\
            utilized in the ``service`` :meth:`store_configs` method.
        hydra_config: The structured :class:`hydra.HydraConf` config\
            used during the ``task`` execution.
        subtask_config: The structured :class:`.BaseSubtaskConfig`\
            config used during the ``subtask`` execution.
    """

    task_config_name: ClassVar[str] = "config"
    task_config_path: ClassVar[str] = (
        sys.argv[0].rsplit("/", maxsplit=1)[0] + "/"
    )
    task_hydra_defaults: ClassVar[list[Any]] = ["_self_", {"task": None}]
    hydra_config: type[BaseHydraConfig] = BaseHydraConfig
    subtask_config: type[BaseSubtaskConfig] = BaseSubtaskConfig

    @final
    @classmethod
    def store_configs_and_run_task(cls: type["BaseTaskRunner"]) -> None:
        """Stores various configs and runs the ``task``.

        Args:
            cls: The :class:`BaseTaskRunner` subclass calling this\
                method.
        """
        store = ZenStore()
        cls.store_configs(store)
        store.add_to_hydra_store()
        zen(cls.run_subtask).hydra_main(
            config_path=cls.task_config_path,
            config_name=cls.task_config_name,
            version_base=None,
        )

    @classmethod
    @abstractmethod
    def store_configs(cls: type["BaseTaskRunner"], store: ZenStore) -> None:
        """Stores structured configs.

        .. warning::

                Make sure to call this method if you are overriding it.

        Stores the :class:`hydra.HydraConf` config.

        Args:
            cls: See :paramref:`~store_configs_and_run_task.cls`.
            store: A :class:`hydra_zen.ZenStore` instance that manages\
                the :mod:`hydra-core` configuration store.
        """
        store(
            cls.hydra_config(
                callbacks={"log_job_return": fs_builds(LogJobReturnCallback)},
            ),
            name="config",
            group="hydra",
        )

    @staticmethod
    @abstractmethod
    def run_subtask(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Runs the ``subtask`` given :paramref:`config`.

        This method is meant to hold the ``subtask`` execution logic.
        """
