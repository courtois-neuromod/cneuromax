""":func:`create_load_function`."""

from collections.abc import Callable

from torch import Tensor

from .load import get_transformed_data_path, load_data
from .overlap import get_overlapping_content_ids
from .paths import KWPredDatasetPaths


def create_load_function(
    paths: KWPredDatasetPaths,
) -> tuple[Callable[[int], dict[str, Tensor]], int]:
    """Creates a function to load data given an index.

    Args:
        paths: See :class:`.KWPredDatasetPaths`.
        seq_len_sec: See :paramref:`~.KWPredDatasetConfig.seq_len_sec`.
        batch_seq_freq_sec: See\
            :paramref:`~.KWPredDataConfig.batch_seq_freq_sec`.

    Returns:
        A function that inputs an index and returns the data for that\
            index + the number of data points.
    """
    content_id_lengths = []
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
        content_id_lengths.append(starting_second - 10)
    total_length = sum(content_id_lengths)
    num_data_points = total_length // 10
    # Create a list of cumulative content ID lengths
    cumulative_content_id_lengths = [0]
    # 0, INT_1, INT_1 + INT_2, INT_1 + INT_2 + INT_3, ...
    for length in content_id_lengths:
        cumulative_content_id_lengths.append(
            cumulative_content_id_lengths[-1] + length,
        )

    def load_fn(idx: int) -> dict[str, Tensor]:
        corresponding_second = idx * 10
        # Find the content ID
        for i, cum_length in enumerate(cumulative_content_id_lengths):
            if corresponding_second < cum_length:
                content_id = overlapping_content_ids[i - 1]
                break
        # Find the starting second
        starting_second = (
            corresponding_second - cumulative_content_id_lengths[i]
        )
        # Load the data
        return load_data(
            paths=paths,
            content_id=content_id,
            starting_second=starting_second,
        )

    return load_fn, num_data_points
