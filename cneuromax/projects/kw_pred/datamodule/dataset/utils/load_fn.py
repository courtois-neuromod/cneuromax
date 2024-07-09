""":func:`create_load_function` and its helper functions."""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from torch import Tensor

from .load import get_transformed_data_path, load_data
from .overlap import get_overlapping_content_ids
from .paths import KWPredDatasetPaths


def create_load_function(
    paths: KWPredDatasetPaths,
    content_id: int | None = None,
) -> tuple[Callable[[int, int], dict[str, Tensor]], int]:
    """Creates a function to load data given an index.

    Args:
        paths: See :class:`.KWPredDatasetPaths`.
        content_id: See :paramref:`.KWPredDatasetConfig.content_id`.

    Returns:
        A function that inputs an index and returns the corresponding\
            data.
    """
    if content_id:
        overlapping_content_ids = [content_id]
    else:
        overlapping_content_ids = get_overlapping_content_ids(paths=paths)
        random.seed(0)
        random.shuffle(overlapping_content_ids)
        logging.info("First 10 content IDs: %s", overlapping_content_ids[:10])
        logging.info("Last 10 content IDs: %s", overlapping_content_ids[-10:])
    data_map = create_data_map(
        paths=paths,
        overlapping_content_ids=overlapping_content_ids,
    )

    def load_fn(
        idx: int,
        num_klk_wav_corners: int,
    ) -> dict[str, Tensor]:

        content_id, starting_time = data_map[idx]
        return load_data(
            paths=paths,
            content_id=content_id,
            starting_time=starting_time,
            num_klk_wav_corners=num_klk_wav_corners,
        )

    return load_fn, len(data_map)


def create_data_map(
    paths: KWPredDatasetPaths,
    overlapping_content_ids: list[int],
) -> list[tuple[int, float]]:
    """Self-explanatory.

    Args:
        paths: See :class:`.KWPredDatasetPaths`.
        overlapping_content_ids: See\
            :func:`.get_overlapping_content_ids`.

    Returns:
        A function that inputs an index and returns the corresponding\
            data.
    """
    data_dir = paths.ae_dir or paths.af_dir or paths.ve_dir
    if not isinstance(data_dir, Path):
        error_msg = "At least one of `ae_dir`, `af_dir`, `ve_dir` is missing."
        raise TypeError(error_msg)
    transformed_data_type = (
        "AE" if paths.ae_dir else "AF" if paths.af_dir else "VE"
    )
    data_map: list[tuple[int, float]] = []
    for content_id in overlapping_content_ids:
        starting_second = 0
        while get_transformed_data_path(
            transformed_data_dir=data_dir,
            transformed_data_type=transformed_data_type,
            content_id=content_id,
            starting_second=starting_second,
        ).exists():
            data_map.append((content_id, starting_second))
            starting_second += 10
    return data_map
