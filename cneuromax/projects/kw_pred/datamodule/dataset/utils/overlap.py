""":func:`get_overlapping_content_ids` and its helper functions."""

from pathlib import Path

from .paths import KWPredDatasetPaths


def get_overlapping_content_ids(paths: KWPredDatasetPaths) -> list[int]:
    """Finds overlapping content IDs between all data sources.

    Args:
        paths: See :class:`.KWPredDatasetPaths`.

    Returns:
        The complete list of content IDs that are present in all three\
            data directories.
    """
    ae_ids = get_transformed_data_content_ids(
        transformed_data_dir=paths.ae_dir,
    )
    af_ids = get_transformed_data_content_ids(
        transformed_data_dir=paths.af_dir,
    )
    ve_ids = get_transformed_data_content_ids(
        transformed_data_dir=paths.ve_dir,
    )
    kw_ids = get_kw_content_ids(kw_dir=paths.kw_dir)
    # Only select the content IDs that are present in all data sources
    overlapping_content_ids = list(
        set(ae_ids) & set(af_ids) & set(ve_ids) & set(kw_ids),
    )
    overlapping_content_ids.sort()
    return overlapping_content_ids


def get_transformed_data_content_ids(transformed_data_dir: Path) -> list[int]:
    """Returns content IDs present in :paramref:`transformed_data_dir`.

    Args:
        transformed_data_dir: Either\
            :paramref:`~.KWPredDatasetPaths.ae_dir`,
            :paramref:`~.KWPredDatasetPaths.af_dir`, or\
            :paramref:`~.KWPredDatasetPaths.ve_dir`.

    Returns:
        List of content IDs found in :paramref:`transformed_data_dir`.
    """
    content_ids = []
    # ..., .../ID2365_XXXX.pt, .../ID2365_XXXX.pt, ...
    for entry in transformed_data_dir.iterdir():
        try:
            # .../ID2365_XXXX.pt -> 2365
            entry_id = int(entry.stem.split("_")[0][2:])
        except ValueError:
            continue
        content_ids.append(entry_id)
    # ..., 2365, 2368, ...
    content_ids = list(set(content_ids))
    content_ids.sort()
    return content_ids


def get_kw_content_ids(kw_dir: Path) -> list[int]:
    """Returns all content IDs present in :paramref:`kw_dir`.

    Args:
        kw_dir: See :paramref:`~.KWPredDatasetPaths.kw_dir`.

    Returns:
        List of content IDs found in :paramref:`kw_dir`.
    """
    content_ids: list[int] = []
    for entry in kw_dir.iterdir():  # ..., .../ID2365/, .../ID2368/, ...
        try:
            entry_id = int(entry.stem[2:])  # .../ID2365/ -> 2365
        except ValueError:
            continue
        content_ids.append(entry_id)
    # ..., 2365, 2368, ...
    return content_ids
