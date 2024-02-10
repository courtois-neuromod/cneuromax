""":func:`load_interpolated_data` and its helper functions."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated as An

import torch
import torch.nn.functional as f
import torchaudio
from jaxtyping import Float32
from torch import Tensor

from cneuromax.utils.beartype import one_of


def create_load_fn(
    paths: dict[str, Path],
    num_10s_segments: int,
) -> Callable[[int], dict[str, Tensor]]:
    """Creates a function to load data given an index.

    Args:
        paths: See :attr:`~.KWPredDataset.paths`.
        num_10s_segments: See\
            :paramref:`~.KWPredDataConfig.num_10s_segments`.

    Returns:
        A function that takes a dictionary of paths and an index and\
            returns the data for that index.
    """
    content_ids_lengths = get_valid_content_ids_lengths(paths=paths)
    data_map = create_data_map(
        paths=paths,
        content_ids_lengths=content_ids_lengths,
        num_10s_segments=num_10s_segments,
    )

    def load_fn(idx: int) -> dict[str, Tensor]:
        return data_map[idx]

    return load_fn


def load_interpolated_data(
    audio_embeddings_dir: Path,
    video_embeddings_dir: Path,
    klk_wavs_dir: Path,
    content_id: int,
) -> dict[str, Tensor]:
    """Loads :paramref:`content_id` data and interpolates to max length.

    Args:
        audio_embeddings_dir: See\
            :paramref:`~.get_valid_content_ids.audio_embeddings_dir`.
        video_embeddings_dir: See\
            :paramref:`~.get_valid_content_ids.video_embeddings_dir`.
        klk_wavs_dir: See \
            :paramref:`~.get_valid_content_ids.klk_wavs_dir`.
        content_id: See :mod:`.kw_pred` terminology.

    Returns:
        The return value of :func:`load_content_id_data` interpolated\
            to the length of the ``.klk`` ``.wav`` data.
    """
    data = load_content_id_data(
        audio_embeddings_dir=audio_embeddings_dir,
        video_embeddings_dir=video_embeddings_dir,
        klk_wavs_dir=klk_wavs_dir,
        content_id=content_id,
    )
    len_klk_wavs_data = data["KW BL"].shape[1]
    for item in ["AE", "VE"]:
        data[item] = f.interpolate(data[item], size=len_klk_wavs_data)
    return data


def load_content_id_data(
    audio_embeddings_dir: Path,
    video_embeddings_dir: Path,
    klk_wavs_dir: Path,
    content_id: int,
) -> dict[str, Tensor]:
    """Fetches data for :paramref:`content_id`.

    Args:
        audio_embeddings_dir: See\
            :paramref:`~.get_valid_content_ids.audio_embeddings_dir`.
        video_embeddings_dir: See\
            :paramref:`~.get_valid_content_ids.video_embeddings_dir`.
        klk_wavs_dir: See\
            :paramref:`~.get_valid_content_ids.klk_wavs_dir`.
        content_id: See :mod:`.kw_pred` terminology.

    Returns:
        A dictionary with keys "AE", "VE", "KW BL", "KW BR", "KW FL"\
            & "KW FR" + the corresponding :class:`torch.Tensor` data.
    """
    # Load all the data
    data_ae, final_ae_second = load_embeddings_data(
        embeddings_dir=audio_embeddings_dir,
        embeddings_type="audio",
        content_id=content_id,
    )
    data_ve, final_ve_second = load_embeddings_data(
        embeddings_dir=video_embeddings_dir,
        embeddings_type="video",
        content_id=content_id,
    )
    data_kw, final_kw_second = load_klk_wav_data(
        klk_wavs_dir=klk_wavs_dir,
        content_id=content_id,
    )
    data: dict[str, Tensor] = {"AE": data_ae, "VE": data_ve, **data_kw}
    # Adjust lengths so that they all match
    min_seconds = min(final_ae_second, final_ve_second, final_kw_second)

    if final_ae_second > min_seconds:
        data["AE"] = data["AE"][: int(min_seconds * 62)]
    if final_ve_second > min_seconds:
        data["VE"] = data["VE"][: int(min_seconds * 241)]
    if final_kw_second > min_seconds:
        pass

    smallest_kw_length = data_kw["BL"].shape[1]
    for pos in ["BL", "BR", "FL", "FR"]:
        data[pos] = data[pos][:smallest_length]

    data.update(klk_wav_data)
    return data


def load_embeddings_data(
    embeddings_dir: Path,
    embeddings_type: An[str, one_of("audio", "video")],
    content_id: int,
) -> tuple[Float32[Tensor, " ..."], int]:
    """Loads the embeddings for :paramref:`content_id`.

    Args:
        embeddings_dir: See\
            :paramref:`~.get_embeddings_data.embeddings_dir`.
        embeddings_type: Either "audio" or "video", used to deal with\
            the different file naming conventions.
        content_id: The ID for the content that is being validated.

    Returns:
        The embeddings for the :paramref:`content_id` and its length in\
            seconds.
    """
    # Naming convention detail
    ms_format = "00" if embeddings_type == "audio" else "0"

    # Helper function to get the file path
    def get_data_file(starting_second: int) -> Path:
        return (
            embeddings_dir / f"ID{content_id}_{starting_second}.{ms_format}_"
            f"{starting_second+10}.{ms_format}.pt"
        )

    # Figure out the total number of files
    max_second = 0
    while get_data_file(starting_second=max_second).exists():
        max_second += 10
    num_files = max_second // 10
    # Initialize the data tensor
    if embeddings_type == "audio":
        data = torch.empty((62 * num_files // 10, 8, 768), dtype=torch.float32)
    else:  # embeddings_type == "video"
        data = torch.empty((241 * num_files // 10, 2048), dtype=torch.float32)
    # Load the data from the files one by one
    second = 0
    while second < max_second:
        if embeddings_type == "audio":
            data_chunk: Float32[Tensor, " 62 8 768"] = torch.load(
                get_data_file(second),
            )
            data[62 * second // 10 : 62 * (second + 10) // 10] = data_chunk
        else:  # embeddings_type == "video"
            data_chunk: Float32[Tensor, " 241 2048"] = torch.load(  # type: ignore[no-redef]
                get_data_file(second),
            )
            data[241 * second // 10 : 241 * (second + 10) // 10] = data_chunk
        second += 10
    return data, max_second


def load_klk_wav_data(
    klk_wavs_dir: Path,
    content_id: int,
) -> tuple[dict[str, Float32[Tensor, " 1 num_samples"]], int]:
    """Loads the KLK WAV files for :paramref:`content_id`.

    Args:
        klk_wavs_dir: See\
            :paramref:`~.get_klk_wav_data.klk_wav_id_folder`.
        content_id: See :paramref:`~.get_embeddings_data.content_id`.

    Returns:
        The KLK WAV data for the :paramref:`content_id` and its length\
            in seconds.
    """
    data: dict[str, Float32[Tensor, " ?num_samples"]] = {}
    klk_wavs_content_id_dir = klk_wavs_dir / f"ID{content_id}/"
    # Load data
    for pos in ["BL", "BR", "FL", "FR"]:
        data_file = klk_wavs_content_id_dir / f"/ID{content_id}_{pos}.wav"
        pos_data_and_sample_rate = torchaudio.load(data_file)
        pos_data: Float32[Tensor, " 1 ?num_samples"] = (
            pos_data_and_sample_rate[0]
        )
        sample_rate = pos_data_and_sample_rate[1]
        pos_data: Float32[Tensor, " ?num_samples"] = pos_data.squeeze(0)
        data[pos] = pos_data
    # Trim the data to the smallest length
    smallest_length = min(
        data["KW BL"].shape[1],
        data["KW BR"].shape[1],
        data["KW FL"].shape[1],
        data["KW FR"].shape[1],
    )
    for pos in ["BL", "BR", "FL", "FR"]:
        data[pos] = data[pos][:smallest_length]
    return data, smallest_length / sample_rate
