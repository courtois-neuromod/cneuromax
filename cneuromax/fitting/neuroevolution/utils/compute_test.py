"""Tests for :mod:`~.neuroevolution.utils.compute`."""

from cneuromax.fitting.neuroevolution.utils.compute import compute_save_points


def compute_save_points_test() -> None:
    """:func:`~.compute_save_points` tests."""
    assert compute_save_points(
        prev_num_gens=0,
        total_num_gens=10,
        save_interval=1,
        save_first_gen=False,
    ) == list(range(1, 11, 1))
    assert compute_save_points(
        prev_num_gens=0,
        total_num_gens=10,
        save_interval=1,
        save_first_gen=True,
    ) == list(range(1, 11, 1))
    assert compute_save_points(
        prev_num_gens=0,
        total_num_gens=10,
        save_interval=2,
        save_first_gen=False,
    ) == list(range(2, 11, 2))
    assert compute_save_points(
        prev_num_gens=0,
        total_num_gens=10,
        save_interval=2,
        save_first_gen=True,
    ) == [1, *list(range(2, 11, 2))]
    assert compute_save_points(
        prev_num_gens=20,
        total_num_gens=10,
        save_interval=2,
        save_first_gen=False,
    ) == list(range(22, 31, 2))
    assert compute_save_points(
        prev_num_gens=20,
        total_num_gens=10,
        save_interval=2,
        save_first_gen=True,
    ) == [21, *list(range(22, 31, 2))]
