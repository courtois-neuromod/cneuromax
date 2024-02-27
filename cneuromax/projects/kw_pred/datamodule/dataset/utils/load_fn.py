""":func:`create_load_function`."""

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
    num_klk_wavs_corners: int,
    duration_second: int,
) -> tuple[Callable[[int], dict[str, Tensor]], int]:
    """Creates a function to load data given an index.

    Args:
        paths: See :class:`.KWPredDatasetPaths`.
        num_klk_wavs_corners: See\
            :paramref:`~.KWPredDatasetConfig.num_klk_wavs_corners`.
        duration_second: See\
            :paramref:`~.KWPredDatasetConfig.duration_second`.

    Returns:
        A function that inputs an index and returns the data for that\
            index + the number of data points.
    """
    overlapping_content_ids = get_overlapping_content_ids(
        paths=paths,
    )
    if paths.ae_dir or paths.af_dir or paths.ve_dir:
        return create_conditional_generation_load_function(
            paths=paths,
            overlapping_content_ids=overlapping_content_ids,
            num_klk_wavs_corners=num_klk_wavs_corners,
        )
    return create_unconditional_generation_load_function(
        paths=paths,
        overlapping_content_ids=overlapping_content_ids,
        num_klk_wavs_corners=num_klk_wavs_corners,
        duration_second=duration_second,
    )


def create_unconditional_generation_load_function(
    paths: KWPredDatasetPaths,
    overlapping_content_ids: list[str],
    num_klk_wavs_corners: int,
    duration_second: int,
) -> tuple[Callable[[int], dict[str, Tensor]], int]:
    """:func:`create_load_function` for unconditional generation.

    Args:
        paths: See :class:`.KWPredDatasetPaths`.
        overlapping_content_ids: See :func:`.get_overlapping_content_ids`.
        num_klk_wavs_corners: See\
            :paramref:`~.KWPredDatasetConfig.num_klk_wavs_corners`.
        duration_second: See\
            :paramref:`~.KWPredDatasetConfig.duration_second`.

    Returns:
        A function that inputs an index and returns the data.
    """
    if not isinstance(paths.an_dir, Path):
        error_msg = "The annotation directory path is missing."
        raise TypeError(error_msg)
    for content_id in overlapping_content_ids:
        # Find the annotation file, we have the content ID
        # but the file name is of type
        # ID1506<...>.csv
        matching_files = paths.an_dir.glob(f"ID{content_id}*.csv")
        if len(list(matching_files)) != 1:
            continue
        file = next(matching_files)
        data_df = pd.read_csv(file)
        # Find all segments
        time_tuples: list[tuple[float, float]] = []
        for row in data_df.iterrows():
            time_on = row[1]["TimeOn"]
            time_off = row[1]["TimeOff"]
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
        # Split the segments into chunks
        chunks: list[tuple[float, float]] = []
        for time_tuple in time_tuples:
            duration = time_tuple[1] - time_tuple[0]
            num_segments = int(duration / duration_second)
            for i in range(num_segments):
                start = time_tuple[0] + i * duration_second
                end = start + duration_second
                chunks.append((start, end))

    def load_fn(idx: int) -> dict[str, Tensor]:
        corresponding_second = idx * 10
        # Find the content ID
        for i, cum_length in enumerate(cumulative_content_ids_lengths):
            if corresponding_second < cum_length:
                content_id = overlapping_content_ids[i - 1]
                break
        # Find the starting second
        starting_second = (
            corresponding_second - cumulative_content_ids_lengths[i]
        )
        # Load the data
        return load_data(
            paths=paths,
            content_id=content_id,
            starting_second=starting_second,
        )

    return load_fn, num_data_points


def create_conditional_generation_load_function(
    paths: KWPredDatasetPaths,
    overlapping_content_ids: list[str],
) -> tuple[Callable[[int], dict[str, Tensor]], int]:

    content_ids_lengths = []
    for content_id in overlapping_content_ids:
        starting_second = 0
        while get_transformed_data_path(
            transformed_data_dir=paths.ae_dir,
            transformed_data_type="AE",
            content_id=content_id,
            starting_second=starting_second,
        ).exists():
            starting_second += 10
        # Since the last one didn't exist, we need to subtract 10
        content_ids_lengths.append(starting_second - 10)
    total_length = sum(content_ids_lengths)
    num_data_points = total_length // 10
    # Create a list of cumulative content ID lengths
    cumulative_content_ids_lengths = [0]
    # 0, INT_1, INT_1 + INT_2, INT_1 + INT_2 + INT_3, ...
    for length in content_ids_lengths:
        cumulative_content_ids_lengths.append(
            cumulative_content_ids_lengths[-1] + length,
        )

    def load_fn(idx: int) -> dict[str, Tensor]:
        corresponding_second = idx * 10
        # Find the content ID
        for i, cum_length in enumerate(cumulative_content_ids_lengths):
            if corresponding_second < cum_length:
                content_id = overlapping_content_ids[i - 1]
                break
        # Find the starting second
        starting_second = (
            corresponding_second - cumulative_content_ids_lengths[i]
        )
        # Load the data
        return load_data(
            paths=paths,
            content_id=content_id,
            starting_second=starting_second,
        )

    return load_fn, num_data_points
