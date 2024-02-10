""":class:`KWPredDataset` & its config/paths dataclasses."""

from dataclasses import dataclass
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from .utils import create_load_data_fn


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
            details on audio STFTs (AS).
        video_embeddings_rel_dir: Relative path with respect to\
            :paramref:`root_data_dir` to the directory containing\
            precomputed video embeddings. See :mod:`.kw_pred` for more\
            details on video embeddings (VE).
        klk_wavs_rel_dir: Relative path with respect to\
            :paramref:`root_data_dir` to the directory containing\
            ``.wav`` files extracted from ``.klk`` files. See\
            :mod:`.kw_pred` for more details on ``.klk`` ``.wav``\
            predictions (KW).
        seq_len_second: Length (in seconds) of the sequence for each\
            data point.
    """

    root_data_dir: str = "/media/DATA/"
    audio_embeddings_rel_dir: str = (
        "transformed_data/audio_embeddings/beats/iter3/"
    )
    audio_stft_rel_dir: str = (
        "transformed_data/stft/47f10f892d824399354c7dbb7cfe0629/"
    )
    video_embeddings_rel_dir: str = (
        "transformed_data/video_embeddings/dinov2/dinov2_vitl14/"
    )
    klk_wavs_rel_dir: str = "HEMC_klk_wavs/"
    seq_len_second: int = 10


@dataclass
class KWPredDatasetPaths:
    """:class:`KWPredDataset` paths.

    Args:
        audio_embeddings_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.audio_embeddings_rel_dir`.
        audio_stft_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.audio_stft_rel_dir`.
        video_embeddings_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.video_embeddings_rel_dir`.
        klk_wavs_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.klk_wavs_rel_dir`.
    """

    audio_embeddings_dir: Path
    audio_stft_dir: Path
    video_embeddings_dir: Path
    klk_wavs_dir: Path


class KWPredDataset(Dataset[dict[str, Tensor]]):
    """:mod:`.kw_pred` :class:`torch.utils.data.Dataset`.

    Args:
        config: See :class:`KWPredDataConfig`.

    Attributes:
        config (:class:`KWPredDataConfig`): See :paramref:`config`.

        content_ids (``list[int]``): List of valid content IDs.
    """

    def __init__(self: "KWPredDataset", config: KWPredDatasetConfig) -> None:
        self.config = config
        self.paths = KWPredPaths(
            audio_embeddings_dir=Path(
                config.root_data_dir + config.audio_embeddings_rel_dir,
            ),
            audio_stft_dir=Path(
                config.root_data_dir + config.audio_stft_rel_dir,
            ),
            video_embeddings_dir=Path(
                config.root_data_dir + config.video_embeddings_rel_dir,
            ),
            klk_wavs_dir=Path(config.root_data_dir + config.klk_wavs_rel_dir),
        )
        self.load_data, self.num_data_points = create_load_data_fn(
            paths=paths,
            num_10s_segments=config.num_10s_segments,
        )

    def __len__(self: "KWPredDataset") -> int:
        """See :meth:`torch.utils.data.Dataset.__len__`."""
        return len(self.num_data_points)

    def __getitem__(self: "KWPredDataset", idx: int) -> dict[str, Tensor]:
        """See :meth:`torch.utils.data.Dataset.__getitem__`."""
        return self.load_data(idx=idx)
