""":class:`BaseTaskRunner`.

Check-out :mod:`cneuromax` for an overview of the terms ``subtask``,
``task``, ``project``, ``service`` and ``interface``.
"""
from typing import Any, final

from hydra_zen import ZenStore, zen

from cneuromax.config import BaseHydraConfig, BaseSubtaskConfig


class BaseTaskRunner:
    """``task`` runner.

    Stores configs and runs the ``task``.

    Attr:
        task_config_name: Name (without the extension) of the\
            ``task`` config file.
        task_config_path: Path to the ``task`` config file relative\
            to the ``project`` root directory.
        hydra_config: The structured :class:`hydra.HydraConf` config\
            used during the ``task`` execution.
        subtask_config: The structured :class:`.BaseSubtaskConfig`\
            config used during the ``subtask`` execution.
    """

    task_config_name: str = "config"
    task_config_path: str = "."
    hydra_config: type[BaseHydraConfig] = BaseHydraConfig
    subtask_config: type[BaseSubtaskConfig] = BaseSubtaskConfig

    @final
    @staticmethod
    def store_configs_and_run_task() -> Any:  # noqa: ANN401
        """Stores various configs and runs the ``task``."""
        store = ZenStore()
        BaseTaskRunner.store_configs(store)
        store.add_to_hydra_store(overwrite_ok=True)
        zen(BaseTaskRunner.validate_subtask_config_and_run).hydra_main(
            config_path=BaseTaskRunner.task_config_path,
            config_name=BaseTaskRunner.task_config_name,
            version_base=None,
        )

    @staticmethod
    def store_configs(store: ZenStore) -> None:
        """Stores ``interface`` :mod:`hydra-core` configs.

        .. warning::

            Make sure to call this method if you are overriding it.

        Args:
            store: A :class:`hydra_zen.ZenStore` instance that manages\
                the :mod:`hydra-core` configuration store.
        """
        store(BaseTaskRunner.hydra_config, name="config", group="hydra")
        store(BaseTaskRunner.subtask_config, name="config")

    @final
    @staticmethod
    def validate_subtask_config_and_run(
        config: BaseSubtaskConfig,
    ) -> Any:  # noqa: ANN401
        """Validates & runs the ``subtask`` given :paramref:`config`.

        Args:
            config: See :attr:`subtask_config`.
        """
        BaseTaskRunner.validate_subtask_config(config)
        BaseTaskRunner.run_subtask(config)

    @staticmethod
    def validate_subtask_config(config: BaseSubtaskConfig) -> None:
        """Validates the structured ``subtask`` :paramref:`config`.

        This method is generally meant to:

        1) Verify that the :paramref:`config` fields are valid given\
           the runtime resources.

        2) Verify that the :paramref:`config` fields do not conflict\
           with each other.

        3) Perform any final operation before the ``subtask`` execution\
           begins.

        Args:
            config: See :attr:`subtask_config`.
        """

    @staticmethod
    def run_subtask(config: BaseSubtaskConfig) -> Any:  # noqa: ANN401
        """Run the ``subtask`` given the :paramref:`config`.

        This method is meant to hold the ``subtask`` execution logic.

        Args:
            config: See :attr:`subtask_config`.
        """
        raise NotImplementedError
