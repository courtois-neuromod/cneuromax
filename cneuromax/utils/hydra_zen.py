""":mod:`hydra-zen` utilities."""
from dataclasses import is_dataclass
from typing import Any

from hydra_zen import make_custom_builds_fn
from hydra_zen.wrapper import default_to_config
from omegaconf import OmegaConf

fs_builds = make_custom_builds_fn(  # type: ignore[var-annotated]
    populate_full_signature=True,
    hydra_convert="partial",
)
""":mod:`hydra-zen` custom build function.

Args:
    populate_full_signature: Set to ``True``.
    hydra_convert: Set to ``"partial"``.
"""
pfs_builds = make_custom_builds_fn(  # type: ignore[var-annotated]
    zen_partial=True,
    populate_full_signature=True,
    hydra_convert="partial",
)
""":mod:`hydra-zen` custom build function.

Args:
    zen_partial: Set to ``True``.
    populate_full_signature: Set to ``True``.
    hydra_convert: Set to ``"partial"``.
"""


def destructure(x: Any) -> Any:  # noqa: ANN401
    """Disables :mod:`hydra` config type checking.

    See `discussion <https://github.com/mit-ll-responsible-ai/\
        hydra-zen/discussions/621#discussioncomment-7938326>`_.
    """
    x = default_to_config(x)  # apply the default auto-config logic of `store`
    if is_dataclass(x):
        # Recursively converts:
        # dataclass -> omegaconf-dict (backed by dataclass types)
        #
        return OmegaConf.create(
            OmegaConf.to_container(
                OmegaConf.create(x),  # type: ignore[call-overload]
            ),
        )
    return x
