""":class:`KWPredDataset` + its config."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated as An

from torch import Tensor
from torch.utils.data import Dataset

from cneuromax.utils.beartype import ge, le

from .utils import KWPredDatasetPaths, create_load_function


@dataclass
class KWPredDatasetConfig:
    """:class:`KWPredDataset` config.

    The default data file structure is of the form:

    .. code-block:: text

        DATA/
        ├── HEMC_klk_wavs/
        │   ├── ...
        │   ├── ID2365/
        │   │   ├── ID2365_BL.wav
        │   │   ├── ID2365_BR.wav
        │   │   ├── ID2365_FL.wav
        │   │   └── ID2365_FR.wav
        │   ├── ...
        │   ├── ID2368/
        │   │   ├── ID2368_BL.wav
        │   │   └── ...
        │   └── ...
        ├─ ISD-Sust-006-Engine/
        │   ├─ ...
        │   ├─ ID2361__Destroyer__ISD-Sust-006-Engine.csv
        │   ├─ ...
        │   ├─ ID2363__Anna__ISD-Sust-006-Engine.csv
        │   └─ ...
        └─ transformed_data/
            ├── audio_embeddings/
            │   └── beats/
            │       └── iter3/
            │           ├── ...
            │           ├── ID2365_0.0_10.0.pt
            │           ├── ID2365_10.0_20.0.pt
            │           ├── ...
            │           ├── ID2368_0.0_10.0.pt
            │           └── ...
            ├── stft/
            │   └── 47f10f892d824399354c7dbb7cfe0629/
            │       ├── ID2360_0_to_10.pt
            │       ├── ID2360_10_to_20.pt
            │       ├── ...
            │       ├── ID2368_0_to_10.pt
            │       └── ...
            └── video_embeddings/
                └── dinov2/
                    └── dinov2_vitl14/
                        ├── ...
                        ├── ID2365_0.0_10.0.pt
                        ├── ID2365_10.0_20.0.pt
                        ├── ...
                        ├── ID2368_0.0_10.0.pt
                        └── ...

    Args:
        root_data_dir: Path to the high-level data directory.
        audio_embeddings_rel_dir: Relative path with respect to\
            :paramref:`root_data_dir` to the directory containing\
            precomputed audio embeddings. See :mod:`.kw_pred` for more\
            details on audio embeddings (AE).
        audio_stft_rel_dir: Relative path with respect to\
            :paramref:`root_data_dir` to the directory containing\
            precomputed transformed audio data through Short-Time\
            Fourier Transform (STFT). See :mod:`.kw_pred` for more\
            details on audio STFTs (AF).
        video_embeddings_rel_dir: Relative path with respect to\
            :paramref:`root_data_dir` to the directory containing\
            precomputed video embeddings. See :mod:`.kw_pred` for more\
            details on video embeddings (VE).
        annot_rel_dir: Relative path with respect to\
            :paramref:`root_data_dir` to the directory containing\
            ``.csv`` files with annotations of the\
            ``ISD-Sust-006-Engine`` category for each movie.
        klk_wavs_rel_dir: Relative path with respect to\
            :paramref:`root_data_dir` to the directory containing\
            ``.wav`` files extracted from ``.klk`` files. See\
            :mod:`.kw_pred` for more details on ``.klk`` ``.wav``\
            predictions (KW).
        num_klk_wavs_corners: Number of corners in the ``.klk``\
            ``.wav`` files to model.
        duration_seconds: Length of each data point in seconds.
    """

    root_data_dir: str = "/media/DATA/"
    audio_embeddings_rel_dir: str | None = (
        "transformed_data/audio_embeddings/beats/iter3/"
    )
    audio_stft_rel_dir: str | None = (
        "transformed_data/stft/47f10f892d824399354c7dbb7cfe0629/"
    )
    video_embeddings_rel_dir: str | None = (
        "transformed_data/video_embeddings/dinov2/dinov2_vitl14/"
    )
    annot_rel_dir: str | None = "ISD-Sust-006-Engine/"
    klk_wavs_rel_dir: str = "HEMC_klk_wavs/"
    num_klk_wavs_corners: An[int, ge(1), le(4)] = 1
    duration_seconds: An[int, ge(1)] = 10


class KWPredDataset(Dataset[dict[str, Tensor]]):
    """:mod:`.kw_pred` :class:`torch.utils.data.Dataset`.

    Args:
        config: See :class:`KWPredDataConfig`.

    Attributes:
        config (:class:`KWPredDataConfig`): See :paramref:`config`.
        load_data_fn (``Callable[[int], dict[str, Tensor]]``): Function\
            to load data given an index.
        num_data_points (``int``): Number of data points in the dataset.
    """

    def __init__(self: "KWPredDataset", config: KWPredDatasetConfig) -> None:
        self.config = config
        paths = KWPredDatasetPaths(
            ae_dir=(
                Path(
                    config.root_data_dir + config.audio_embeddings_rel_dir,
                )
                if config.audio_embeddings_rel_dir
                else None
            ),
            af_dir=(
                Path(
                    config.root_data_dir + config.audio_stft_rel_dir,
                )
                if config.audio_stft_rel_dir
                else None
            ),
            ve_dir=(
                Path(
                    config.root_data_dir + config.video_embeddings_rel_dir,
                )
                if config.video_embeddings_rel_dir
                else None
            ),
            an_dir=(
                Path(config.root_data_dir + config.annot_rel_dir)
                if config.annot_rel_dir
                else None
            ),
            kw_dir=(Path(config.root_data_dir + config.klk_wavs_rel_dir)),
        )
        self.load_data_fn, self.num_data_points = create_load_function(
            paths=paths,
            duration_second=config.duration_seconds,
        )

    def __len__(self: "KWPredDataset") -> int:
        """See :meth:`torch.utils.data.Dataset.__len__`."""
        return self.num_data_points

    def __getitem__(self: "KWPredDataset", idx: int) -> dict[str, Tensor]:
        """See :meth:`torch.utils.data.Dataset.__getitem__`."""
        while True:  # spooky (~'o')~ ...
            try:
                data: dict[str, Tensor] = self.load_data_fn(
                    idx,
                    self.config.duration_seconds,
                    self.config.num_klk_wavs_corners,
                )
                if data["KW BL"].min() == data["KW BL"].max():
                    raise Exception  # noqa: TRY301, TRY002
                return data  # noqa: TRY300
            except Exception:  # noqa: PERF203, BLE001
                idx = (idx * 2) % self.num_data_points
