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
    ids: list[list[int]] = [
        get_data_content_ids(data_dir=path) for path in paths if path
    ]
    # Only select the content IDs that are present in all data sources
    overlapping_content_ids = (
        # >>> a = [1,2,3,4,5]
        # >>> b = [2,3,4,5,6]
        # >>> c = [3,4,5,6,7]
        # >>> l = [a,b,c]
        # >>> set(l[0]).intersection(*l[1:]) -> {3, 4, 5}
        list(set(ids[0]).intersection(*ids[1:]))
        if len(ids) > 1
        else ids[0]
    )
    overlapping_content_ids.sort()
    return overlapping_content_ids


def get_data_content_ids(data_dir: Path) -> list[int]:
    """Returns content IDs present in :paramref:`data_dir`.

    Args:
        data_dir: Any `dir` entry in :class:`.KWPredDatasetPaths`.

    Returns:
        List of content IDs found in :paramref:`transformed_data_dir`.
    """
    content_ids = []
    # `ae_dir`, `af_dir`, `ve_dir`:
    # ..., .../ID2365_XXXX.pt, .../ID2365_XXXX.pt, ...
    # `an_dir`:
    # ..., ID2361__Destroyer__ISD-Sust-006-Engine.csv, ...
    # ID2363__Anna__ISD-Sust-006-Engine.csv, ...
    # `kw_dir`:
    # ..., ID2365/, ID2368/, ...
    for entry in data_dir.iterdir():
        try:
            # .../ID2365_XXXX.pt -> 2365
            entry_id = int(entry.stem[2:6])
        except ValueError:
            continue
        content_ids.append(entry_id)
    # For `ae_dir`, `af_dir`, `ve_dir`
    content_ids = list(set(content_ids))
    content_ids.sort()
    return content_ids
