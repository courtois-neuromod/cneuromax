""":class:`.BaseTaskRunner`."""

from abc import ABC, abstractmethod
from typing import Any, final

from hydra_zen import ZenStore, zen
from omegaconf import OmegaConf

from cneuromax.config import BaseHydraConfig
from cneuromax.utils.hydra_zen import destructure
from cneuromax.utils.runner import (
    get_absolute_project_path,
    get_project_and_task_names,
)
from cneuromax.utils.wandb import login_wandb


class BaseScheduleRunner(ABC):
    """Starts the ``schedule`` ``run``(s)."""


class BaseTaskRunner(ABC):
    """Starts the ``task`` ``run``(s)."""

    hydra_config = BaseHydraConfig

    @final
    @classmethod
    def store_configs_and_start_runs(cls: type["BaseTaskRunner"]) -> None:
        """Stores various configs and starts the ``run``(s).

        Args:
            cls: The :class:`BaseTaskRunner` subclass calling this
                method.
        """
        OmegaConf.register_new_resolver("eval", eval)
        store = ZenStore()
        cls.store_configs(store=store)
        login_wandb()
        zen(cls.run).hydra_main(
            config_path=get_absolute_project_path(),
            config_name="config",
            version_base=None,
        )

    @classmethod
    def store_configs(cls: type["BaseTaskRunner"], store: ZenStore) -> None:
        """Stores structured configs.

        Args:
            cls: See :paramref:`~store_configs_and_run_task.cls`.
            store: A :class:`hydra_zen.ZenStore` instance that manages
                the `Hydra <https://hydra.cc>`_ configuration store.
        """
        store(cls.hydra_config, name="config", group="hydra")
        project_name, task_name = get_project_and_task_names()
        store({"project": project_name}, name="project")
        store({"task": task_name}, name="task")
        # Hydra runtime type checking issues with structured configs:
        # https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/621#discussioncomment-7938326
        # `destructure` disables Hydra's runtime type checking, which is
        # fine since we use Beartype throughout the codebase.
        store = store(to_config=destructure)
        store.add_to_hydra_store(overwrite_ok=True)

    @staticmethod
    @abstractmethod
    def run(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Starts the ``run`` given :paramref:`config`.

        This method is meant to hold the ``run`` execution logic.
        """
