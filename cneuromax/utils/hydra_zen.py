""":doc:`hydra-zen <hydra-zen:index>` utilities."""

from dataclasses import is_dataclass
from typing import Any

from hydra_zen import make_custom_builds_fn
from hydra_zen.wrapper import default_to_config
from omegaconf import OmegaConf

builds = make_custom_builds_fn(  # type: ignore[var-annotated]
    populate_full_signature=False,
    hydra_convert="partial",
)
""":doc:`hydra-zen <hydra-zen:index>` custom build function.

Args:
    populate_full_signature: Set to ``False``.
    hydra_convert: Set to ``"partial"``.
"""

p_builds = make_custom_builds_fn(  # type: ignore[var-annotated]
    zen_partial=True,
    populate_full_signature=False,
    hydra_convert="partial",
)
""":doc:`hydra-zen <hydra-zen:index>` custom build function.

Args:
    zen_partial: Set to ``True``.
    populate_full_signature: Set to ``False``.
    hydra_convert: Set to ``"partial"``.
"""

fs_builds = make_custom_builds_fn(  # type: ignore[var-annotated]
    populate_full_signature=True,
    hydra_convert="partial",
)
""":doc:`hydra-zen <hydra-zen:index>` custom build function.

Args:
    populate_full_signature: Set to ``True``.
    hydra_convert: Set to ``"partial"``.
"""

pfs_builds = make_custom_builds_fn(  # type: ignore[var-annotated]
    zen_partial=True,
    populate_full_signature=True,
    hydra_convert="partial",
)
""":doc:`hydra-zen <hydra-zen:index>` custom build function.

Args:
    zen_partial: Set to ``True``.
    populate_full_signature: Set to ``True``.
    hydra_convert: Set to ``"partial"``.
"""


def destructure(x: Any) -> Any:  # noqa: ANN401
    """Disables `Hydra <https://hydra.cc>`_ config type checking.

    See `discussion <https://github.com/mit-ll-responsible-ai/\
        hydra-zen/discussions/621#discussioncomment-7938326>`_.
    """
    # apply the default auto-config logic of `store`
    x = default_to_config(target=x)
    if is_dataclass(obj=x):
        # Recursively converts:
        # dataclass -> omegaconf-dict (backed by dataclass types)
        return OmegaConf.create(
            obj=OmegaConf.to_container(
                cfg=OmegaConf.create(obj=x),  # type: ignore[call-overload]
            ),
        )
    return x
