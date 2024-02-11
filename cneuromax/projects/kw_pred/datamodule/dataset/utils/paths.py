""":class:`KWPredDatasetPaths`."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class KWPredDatasetPaths:
    """:class:`KWPredDataset` paths.

    Args:
        ae_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.audio_embeddings_rel_dir`.
        af_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.audio_stft_rel_dir`.
        ve_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.video_embeddings_rel_dir`.
        kw_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.klk_wavs_rel_dir`.
    """

    ae_dir: Path
    af_dir: Path
    ve_dir: Path
    kw_dir: Path
