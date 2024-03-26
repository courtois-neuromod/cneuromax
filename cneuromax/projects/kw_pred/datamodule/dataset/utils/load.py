""":func:`load_data` and its helper functions."""

from pathlib import Path

import torch
import torch.nn.functional as f
import torchaudio
from einops import rearrange
from jaxtyping import Float32
from torch import Tensor

from .paths import KWPredDatasetPaths


def load_data(
    paths: KWPredDatasetPaths,
    content_id: int,
    starting_time: float,
    duration_second: int,
    num_klk_wav_corners: int,
) -> dict[str, Tensor]:
    """Loads :paramref:`duration_second` seconds of data.

    Args:
        paths: See :class:`.KWPredDatasetPaths`.
        content_id: See :mod:`.kw_pred` terminology.
        starting_time: Self-explanatory.
        duration_second: See\
            :paramref:`~.KWPredDatasetConfig.duration_second`.
        num_klk_wav_corners: See\
            :paramref:`~.KWPredDatasetConfig.num_klk_wav_corners`.

    Returns:
        A dictionary mapping from strings to :class:`torch.Tensor`\
            data. For conditional generation, the keys are a subset of\
            the keys "AE", "AF", "VE" and a subset of "KW BL", "KW BR",\
            "KW FL" & "KW FR". For unconditional generation, the keys\
            are a subset of "KW BL", "KW BR", "KW FL" & "KW FR".
    """
    transformed_data_dict = {}
    if paths.ae_dir:
        ae_data: Float32[Tensor, " num_ae_samples num_ae_1 num_ae_2"] = (
            load_transformed_data(
                transformed_data_dir=paths.ae_dir,
                transformed_data_type="AE",
                content_id=content_id,
                starting_second=int(starting_time),
            )
        )
        # Average out the second dimension.
        ae_data: Float32[Tensor, " num_ae_samples num_ae"] = torch.mean(
            input=ae_data,
            dim=1,
        )
        transformed_data_dict["AE"] = ae_data
    if paths.af_dir:
        af_data: Float32[Tensor, " num_af_samples num_af"] = (
            load_transformed_data(
                transformed_data_dir=paths.af_dir,
                transformed_data_type="AF",
                content_id=content_id,
                starting_second=int(starting_time),
            )
        )
        transformed_data_dict["AF"] = af_data
    if paths.ve_dir:
        ve_data: Float32[Tensor, " num_ve_samples num_ve"] = (
            load_transformed_data(
                transformed_data_dir=paths.ve_dir,
                transformed_data_type="VE",
                content_id=content_id,
                starting_second=int(starting_time),
            )
        )
        transformed_data_dict["VE"] = ve_data

    kw_data: dict[
        str,
        Float32[Tensor, " 4000"] | Float32[Tensor, " dur_sec*400"],
    ] = load_kw_data(
        kw_dir=paths.kw_dir,
        content_id=content_id,
        starting_time=starting_time,
        duration_second=duration_second,
        num_klk_wav_corners=num_klk_wav_corners,
    )
    return {**transformed_data_dict, **kw_data}


def get_transformed_data_path(
    transformed_data_dir: Path,
    transformed_data_type: str,
    content_id: int,
    starting_second: int,
) -> Path:
    """Gets the path to the transformed data.

    Args:
        transformed_data_dir: See\
            :paramref:`~.get_transformed_data_content_ids.transformed_data_dir`.
        transformed_data_type: Used to deal with the different file\
            naming conventions.
        content_id: See :mod:`.kw_pred` terminology.
        starting_second: See :paramref:`~load_data.starting_second`.

    Returns:
        The path to the transformed data for the :paramref:`content_id`\
            from :paramref:`starting_second` to\
            :paramref:`starting_second` + 10.
    """
    # Naming convention detail
    if transformed_data_type == "AE":
        ms_format = ".00"
    elif transformed_data_type == "AF":
        ms_format = ""
    else:  # transformed_data_type == "VE"
        ms_format = ".0"

    return transformed_data_dir / (
        f"ID{content_id}_{starting_second}.{ms_format}_"
        f"{starting_second+10}.{ms_format}.pt"
    )


def load_transformed_data(
    transformed_data_dir: Path,
    transformed_data_type: str,
    content_id: int,
    starting_second: int,
) -> (
    Float32[Tensor, " num_samples num_features"]
    | Float32[Tensor, " num_samples num_features_1 num_features_2"]
):
    """Loads 10 seconds of transformed data.

    Args:
        transformed_data_dir: See\
            :paramref:`~.get_transformed_data_content_ids.transformed_data_dir`.
        transformed_data_type: Used to deal with the different file\
            naming conventions.
        content_id: See :mod:`.kw_pred` terminology.
        starting_second: See :paramref:`~load_data.starting_second`.

    Returns:
        The transformed data for the :paramref:`content_id` from\
            :paramref:`starting_second` to :paramref:`starting_second`\
            + 10.
    """
    data: (
        Float32[Tensor, " num_samples num_features"]
        | Float32[Tensor, " num_samples num_features_1 num_features_2"]
    ) = torch.load(
        get_transformed_data_path(
            transformed_data_dir=transformed_data_dir,
            transformed_data_type=transformed_data_type,
            content_id=content_id,
            starting_second=starting_second,
        ),
    )
    return data


def interpolate_transformed_data(
    data: Float32[Tensor, " num_samples num_features"],
) -> Float32[Tensor, " 4000 data_dim"]:
    """Interpolates the 10 second data to 400 Hz.

    Args:
        data: The data to interpolate.

    Returns:
        The interpolated data.
    """
    data: Float32[Tensor, " 1 num_features num_samples"] = rearrange(
        tensor=data,
        pattern="num_samples num_features -> 1 num_features num_samples",
    )
    data: Float32[Tensor, " 1 num_features 4000"] = f.interpolate(
        input=data,
        size=4000,
    )
    return rearrange(
        tensor=data,
        pattern="1 num_features 4000 -> 4000 num_features",
    )


def load_kw_data(
    kw_dir: Path,
    content_id: int,
    starting_time: float,
    duration_second: int,
    num_klk_wav_corners: int,
) -> dict[str, Float32[Tensor, " num_samples"]]:
    """Loads ``.klk`` ``.wav`` data.

    Args:
        kw_dir: See :paramref:`~.KWPredDatasetPaths.kw_dir`.
        content_id: See :mod:`.kw_pred` terminology.
        starting_time: See :paramref:`~load_data.starting_time`.
        duration_second: See\
            :paramref:`~.KWPredDatasetConfig.duration_second`.
        num_klk_wav_corners: See\
            :paramref:`~.KWPredDatasetConfig.num_klk_wav_corners`.

    Returns:
        The ``.klk`` ``.wav`` data for the :paramref:`content_id` from\
            :paramref:`starting_time` to :paramref:`starting_time`\
            + :paramref:`duration_second`.
    """
    data: dict[str, Float32[Tensor, " num_samples"]] = {}
    kw_content_id_dir = kw_dir / f"ID{content_id}/"
    # Load data
    corners = ["BL", "BR", "FL", "FR"][:num_klk_wav_corners]
    for pos in corners:
        data_file = kw_content_id_dir / f"ID{content_id}_{pos}.wav"
        pos_data, _ = torchaudio.load(
            uri=data_file,
            frame_offset=int(starting_time * 400),
            num_frames=duration_second * 400,
        )
        pos_data: Float32[Tensor, " num_samples"] = pos_data.squeeze(0)
        data[f"KW {pos}"] = pos_data
    return data
