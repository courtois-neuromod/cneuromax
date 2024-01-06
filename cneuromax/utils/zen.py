""":mod:`hydra-zen` utilities."""
from collections.abc import Callable
from typing import Any

from hydra_zen import ZenStore, make_custom_builds_fn
from omegaconf import MISSING

fs_builds = make_custom_builds_fn(  # type: ignore[var-annotated]
    populate_full_signature=True,
    hydra_convert="partial",
)
pfs_builds = make_custom_builds_fn(  # type: ignore[var-annotated]
    zen_partial=True,
    populate_full_signature=True,
    hydra_convert="partial",
)


def store_wandb_logger_configs(
    store: ZenStore,
    clb: Callable[..., Any],
) -> None:
    """Stores :mod:`hydra-core` ``logger`` group configs.

    Config names: ``wandb``, ``wandb_simexp``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        clb: :mod:`wandb` initialization callable.
    """
    return
    base_args: dict[str, Any] = {  # `fs_builds`` does not like dict[str, str]
        "name": "{hydra:task_name}",
        "save_dir": "${data_dir}",
        "project": "{hydra:project_name}",
    }
    store(
        fs_builds(clb, **base_args, entity=MISSING),
        group="logger",
        name="wandb",
    )
    store(
        fs_builds(clb, **base_args, entity="cneuroml"),
        group="logger",
        name="wandb_simexp",
    )
