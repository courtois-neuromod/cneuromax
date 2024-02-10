""":func:`get_valid_content_ids` and its helper functions."""

import logging
from pathlib import Path

from .load import load_content_id_data


def get_valid_content_ids_lengths(paths) -> list[int]:
    """Returns a list of content IDs suitable for training.

    Args:
        audio_embeddings_dir: Concatenation of\
            :paramref:`~.KWPredDataConfig.root_data_dir` and\
            :paramref:`~.KWPredDataConfig.audio_embeddings_rel_dir`.
        video_embeddings_dir: Concatenation of\
            :paramref:`~.KWPredDataConfig.root_data_dir` and\
            :paramref:`~.KWPredDataConfig.video_embeddings_rel_dir`.
        stft_dir: Concatenation of\
            :paramref:`~.KWPredDataConfig.root_data_dir` and\
            :paramref:`~.KWPredDataConfig.stft_rel_dir`.
        klk_wavs_dir: Concatenation of\
            :paramref:`~.KWPredDataConfig.root_data_dir` and\
            :paramref:`~.KWPredDataConfig.klk_wavs_rel_dir`.

    Returns:
        A list of content IDs that have all required data to be used in\
            training.
    """
    content_ids = get_overlapping_content_ids(
        audio_embeddings_dir=audio_embeddings_dir,
        video_embeddings_dir=video_embeddings_dir,
        stft_dir=stft_dir,
        klk_wavs_dir=klk_wavs_dir,
    )
    return get_no_error_content_ids(
        audio_embeddings_dir=audio_embeddings_dir,
        video_embeddings_dir=video_embeddings_dir,
        klk_wavs_dir=klk_wavs_dir,
        content_ids=content_ids,
    )


def get_overlapping_content_ids(
    audio_embeddings_dir: Path,
    video_embeddings_dir: Path,
    klk_wavs_dir: Path,
) -> list[int]:
    """Finds overlapping content IDs between all data sources.

    Args:
        audio_embeddings_dir: See\
            :paramref:`~.get_valid_content_ids.audio_embeddings_dir`.
        video_embeddings_dir: See\
            :paramref:`~.get_valid_content_ids.video_embeddings_dir`.
        klk_wavs_dir: See\
            :paramref:`~.get_valid_content_ids.klk_wavs_dir`.

    Returns:
        The complete list of content IDs that are present in all three\
            data directories.
    """
    audio_embeddings_ids = get_embeddings_content_ids(
        embeddings_dir=audio_embeddings_dir,
    )
    video_embeddings_ids = get_embeddings_content_ids(
        embeddings_dir=video_embeddings_dir,
    )
    klk_wavs_ids = get_klk_wav_content_ids(klk_wavs_dir=klk_wavs_dir)
    overlapping_content_ids = list(
        set(audio_embeddings_ids)
        & set(video_embeddings_ids)
        & set(klk_wavs_ids),
    )
    overlapping_content_ids.sort()
    return overlapping_content_ids


def get_embeddings_content_ids(embeddings_dir: Path) -> list[int]:
    """Returns all content IDs present in :paramref:`embeddings_dir`.

    Args:
        embeddings_dir: Either\
            :paramref:`~get_valid_content_ids.audio_embeddings_dir` or\
            :paramref:`~get_valid_content_ids.video_embeddings_dir`.

    Returns:
        List of content IDs found in :paramref:`embeddings_dir`.
    """
    content_ids = []
    # ..., .../ID2365_0.00_10.00.pt, .../ID2365_10.00_20.00.pt, ...
    for entry in embeddings_dir.iterdir():
        try:
            # .../ID2365_0.00_10.00.pt -> 2365
            entry_id = int(entry.stem.split("_")[0][2:])
        except ValueError:
            continue
        content_ids.append(entry_id)
    # ..., 2365, 2368, ...
    content_ids = list(set(content_ids))
    content_ids.sort()
    return content_ids


def get_klk_wav_content_ids(klk_wavs_dir: Path) -> list[int]:
    """Returns all content IDs present in :paramref:`klk_wavs_dir`.

    Args:
        klk_wavs_dir: See\
            :paramref:`~get_valid_content_ids.klk_wavs_dir`.

    Returns:
        List of content IDs found in :paramref:`klk_wavs_dir`.
    """
    content_ids: list[int] = []
    for entry in klk_wavs_dir.iterdir():  # ..., .../ID2365/, .../ID2368/, ...
        try:
            entry_id = int(entry.stem[2:])  # .../ID2365/ -> 2365
        except ValueError:
            continue
        content_ids.append(entry_id)
    # ..., 2365, 2368, ...
    return content_ids


def get_no_error_content_ids(
    audio_embeddings_dir: Path,
    video_embeddings_dir: Path,
    klk_wavs_dir: Path,
    content_ids: list[int],
) -> list[int]:
    """Returns the :paramref:`content_ids` subset that can be loaded.

    Args:
        audio_embeddings_dir: See\
            :paramref:`~get_valid_content_ids.audio_embeddings_dir`.
        video_embeddings_dir: See\
            :paramref:`~get_valid_content_ids.video_embeddings_dir`.
        klk_wavs_dir: See\
            :paramref:`~get_valid_content_ids.klk_wavs_dir`.
        content_ids: See return value of\
            :func:`get_overlapping_content_ids`.

    Raises:
        Exception: Whenever a content ID cannot be properly loaded.

    Returns:
        List of content IDs that do not raise any exceptions when\
            loaded.
    """
    no_error_content_ids = []
    for content_id in content_ids:
        try:
            load_content_id_data(
                audio_embeddings_dir=audio_embeddings_dir,
                video_embeddings_dir=video_embeddings_dir,
                klk_wavs_dir=klk_wavs_dir,
                content_id=content_id,
            )
        except Exception:  # noqa: BLE001
            logging.info(f"Skipping ID{content_id}.")
            continue
        no_error_content_ids.append(content_id)
    return no_error_content_ids
