"""Deep net training.

``config.yaml``:

.. highlight:: yaml
.. code-block:: yaml

    hydra:
        searchpath:
            - file://${oc.env:CNEUROMAX_PATH}/cneuromax/

    defaults:
        - deep_learning_fitting
        - trainer: base
        - litmodule/scheduler: constant
        - litmodule/optimizer: adamw
        - logger: wandb
        - _self_
        - base_config
"""

import hydra
from hydra.core.config_store import ConfigStore
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.trainer import Trainer
from omegaconf import DictConfig
from torch.optim import SGD, Adam, AdamW
from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

from cneuromax.config import process_config
from cneuromax.fitting import store_base_fitting_configs
from cneuromax.fitting.deeplearning.config import (
    DeepLearningFittingHydraConfig,
    post_process_deep_learning_fitting_config,
    pre_process_deep_learning_fitting_config,
)
from cneuromax.fitting.deeplearning.fit import fit
from cneuromax.fitting.deeplearning.nnmodule import store_nnmodule_configs
from cneuromax.utils.hydra import fs_builds, pfs_builds
from cneuromax.utils.wandb import store_logger_configs

__all__ = ["run", "store_configs"]


@hydra.main(config_name="config", config_path=".", version_base=None)
def run(config: DictConfig) -> None:
    """Processes the :mod:`hydra-core` config and fits w/ Deep Learning.

    Args:
        config: See :paramref:`~.pre_process_base_config.config`.
    """
    pre_process_deep_learning_fitting_config(config)
    config = process_config(
        config=config,
        structured_config_class=DeepLearningFittingHydraConfig,
    )
    post_process_deep_learning_fitting_config(config)
    fit(config)


def store_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` Deep Learning fitting configs."""
    store_base_fitting_configs(cs)
    store_logger_configs(cs, clb=WandbLogger)
    store_nnmodule_configs(cs)
    store_optimizer_configs(cs)
    store_scheduler_configs(cs)
    store_trainer_configs(cs)
    cs.store(
        name="deep_learning_fitting",
        node=DeepLearningFittingHydraConfig,
    )


def store_optimizer_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/optimizer`` group configs.

    Config names: ``adam``, ``adamw``, ``sgd``.

    Args:
        cs: See :paramref:`~.store_project_configs.cs`.
    """
    cs.store(group="litmodule/optimizer", name="adam", node=pfs_builds(Adam))
    cs.store(group="litmodule/optimizer", name="adamw", node=pfs_builds(AdamW))
    cs.store(group="litmodule/optimizer", name="sgd", node=pfs_builds(SGD))


def store_scheduler_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/scheduler`` group configs.

    Config names: ``constant``, ``linear_warmup``.

    Args:
        cs: See :paramref:`~.store_project_configs.cs`.
    """
    cs.store(
        group="litmodule/scheduler",
        name="constant",
        node=pfs_builds(get_constant_schedule),
    )
    cs.store(
        group="litmodule/scheduler",
        name="linear_warmup",
        node=pfs_builds(get_constant_schedule_with_warmup),
    )


def store_trainer_configs(cs: ConfigStore) -> None:
    """Stores :mod:`hydra-core` ``trainer`` group configs.

    Config name: ``base``.

    Args:
        cs: See :paramref:`~.store_project_configs.cs`.
    """
    cs.store(
        group="trainer",
        name="base",
        node=fs_builds(
            Trainer,
            accelerator="${device}",
            default_root_dir="${data_dir}/lightning/",
        ),
    )
