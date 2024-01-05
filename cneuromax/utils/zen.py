""":mod:`hydra-zen` utilities."""
from hydra_zen import make_custom_builds_fn

fs_builds = make_custom_builds_fn(  # type: ignore[var-annotated]
    populate_full_signature=True,
    hydra_convert="partial",
)
pfs_builds = make_custom_builds_fn(  # type: ignore[var-annotated]
    zen_partial=True,
    populate_full_signature=True,
    hydra_convert="partial",
)
