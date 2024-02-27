""":func:`create_load_function` and its helper functions."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from torch import Tensor

from .load import get_transformed_data_path, load_data
from .overlap import get_overlapping_content_ids
from .paths import KWPredDatasetPaths


def create_load_function(
    paths: KWPredDatasetPaths,
    duration_second: int,
) -> tuple[Callable[[int, int, int], dict[str, Tensor]], int]:
    """Creates a function to load data given an index.

    Args:
        paths: See :class:`.KWPredDatasetPaths`.
        num_klk_wavs_corners: See\
            :paramref:`~.KWPredDatasetConfig.num_klk_wavs_corners`.
        duration_second: See\
            :paramref:`~.KWPredDatasetConfig.duration_second`.

    Returns:
        A function that inputs an index and returns the corresponding\
            data.
    """
    overlapping_content_ids = get_overlapping_content_ids(
        paths=paths,
    )
    data_map = (
        create_conditional_generation_data_map(
            paths=paths,
            overlapping_content_ids=overlapping_content_ids,
            duration_second=duration_second,
        )
        if paths.ae_dir or paths.af_dir or paths.ve_dir
        else create_unconditional_generation_data_map(
            paths=paths,
            overlapping_content_ids=overlapping_content_ids,
            duration_second=duration_second,
        )
    )

    def load_fn(
        idx: int,
        duration_second: int,
        num_klk_wav_corners: int,
    ) -> dict[str, Tensor]:

        content_id, starting_time = data_map[idx]
        return load_data(
            paths=paths,
            content_id=content_id,
            starting_time=starting_time,
            duration_second=duration_second,
            num_klk_wav_corners=num_klk_wav_corners,
        )

    return load_fn, len(data_map)


def create_unconditional_generation_data_map(
    paths: KWPredDatasetPaths,
    overlapping_content_ids: list[int],
    duration_second: int,
) -> list[tuple[int, float]]:
    """:func:`create_load_function` for unconditional generation.

    Args:
        paths: See :class:`.KWPredDatasetPaths`.
        overlapping_content_ids: See\
            :func:`.get_overlapping_content_ids`.
        num_klk_wavs_corners: See\
            :paramref:`~.KWPredDatasetConfig.num_klk_wavs_corners`.
        duration_second: See\
            :paramref:`~.KWPredDatasetConfig.duration_second`.

    Returns:
        A list of tuples, each containing a content ID and a starting\
            second.
    """
    if not isinstance(paths.an_dir, Path):
        error_msg = "`an_dir` is missing."
        raise TypeError(error_msg)
    data_map: list[tuple[int, float]] = []
    for content_id in overlapping_content_ids:
        # Find and load the annotation file
        matching_files = paths.an_dir.glob(f"ID{content_id}*.csv")
        if len(list(matching_files)) != 1:
            continue
        file = next(matching_files)
        data_df = pd.read_csv(file)
        # Find all annotation blocks
        time_tuples: list[tuple[float, float]] = []
        for row in data_df.iterrows():
            time_on = row[1]["TimeOn"]
            time_off = row[1]["TimeOff"]
            # If an annotation block is immediately followed by another
            # annotation block, we merge them
            if len(time_tuples) > 0 and np.allclose(
                a=time_on,
                b=time_tuples[-1][1],
                # >>> np.allclose(0.93301, 0.9318,atol=1.1e-3) -> False
                # >>> np.allclose(0.93301, 0.93199,atol=1.1e-3) -> True
                atol=1.1e-3,
            ):
                time_tuples[-1] = (time_tuples[-1][0], time_off)
                continue
            time_tuples.append((time_on, time_off))
        for time_tuple in time_tuples:
            duration = time_tuple[1] - time_tuple[0]
            num_segments = int(duration / duration_second)
            for i in range(num_segments):
                start = time_tuple[0] + i * duration_second
                data_map.append((content_id, start))

    return data_map


def create_conditional_generation_data_map(
    paths: KWPredDatasetPaths,
    overlapping_content_ids: list[int],
    duration_second: int,
) -> list[tuple[int, float]]:
    """Self-explanatory.

    Args:
        paths: See :class:`.KWPredDatasetPaths`.
        overlapping_content_ids: See\
            :func:`.get_overlapping_content_ids`.
        duration_second: See\
            :paramref:`~.KWPredDatasetConfig.duration_second`.

    Returns:
        See return value of\
            :func:`.create_unconditional_generation_data_map`.
    """
    data_dir = paths.ae_dir or paths.af_dir or paths.ve_dir
    if not isinstance(dir, Path):
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
            starting_second += duration_second
    return data_map
