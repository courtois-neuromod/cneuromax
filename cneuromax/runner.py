""":class:`BaseTaskRunner`.

Check-out :mod:`cneuromax` for an overview of the terms ``subtask``,
``task``, ``project``, ``service`` and ``interface``.
"""
import sys
from abc import ABC, abstractmethod
from typing import Any, ClassVar, final

from hydra_zen import ZenStore, zen

from cneuromax.config import BaseHydraConfig  # , BaseTaskConfig
from cneuromax.utils.hydra_zen import destructure


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
        task_config: The structured ``task`` config.
    """

    task_config_name: ClassVar[str] = "config"
    task_config_path: ClassVar[str] = (
        sys.argv[0].rsplit("/", maxsplit=1)[0] + "/"
    )
    hydra_config = BaseHydraConfig

    @final
    @classmethod
    def store_configs_and_run_task(cls: type["BaseTaskRunner"]) -> None:
        """Stores various configs and runs the ``task``.

        Args:
            cls: The :class:`BaseTaskRunner` subclass calling this\
                method.
        """
        store = ZenStore()
        store(cls.hydra_config, name="config", group="hydra")
        store = store(to_config=destructure)
        cls.store_configs(store)
        store.add_to_hydra_store(overwrite_ok=True)
        zen(cls.run_subtask).hydra_main(
            config_path=cls.task_config_path,
            config_name=cls.task_config_name,
            version_base=None,
        )

    @classmethod
    @abstractmethod
    def store_configs(cls: type["BaseTaskRunner"], store: ZenStore) -> None:
        """Stores structured configs.

        Stores the :class:`hydra.HydraConf` config.

        Args:
            cls: See :paramref:`~store_configs_and_run_task.cls`.
            store: A :class:`hydra_zen.ZenStore` instance that manages\
                the :mod:`hydra-core` configuration store.
        """

    @staticmethod
    @abstractmethod
    def run_subtask(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Runs the ``subtask`` given :paramref:`config`.

        This method is meant to hold the ``subtask`` execution logic.
        """
