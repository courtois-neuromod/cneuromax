""":class:`KWPredDatasetPaths`."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KWPredDatasetPaths:
    """:class:`KWPredDataset` paths.

    Args:
        ae_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.audio_embeddings_rel_dir`\
            or :obj:`None`.
        af_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.audio_stft_rel_dir`\
            or :obj:`None`.
        ve_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.video_embeddings_rel_dir`\
            or :obj:`None`.
        kw_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.klk_wavs_rel_dir`.
    """

    ae_dir: Path | None
    af_dir: Path | None
    ve_dir: Path | None
    kw_dir: Path

    def __iter__(self: "KWPredDatasetPaths") -> Iterator[Path | None]:
        """See return.

        Returns:
            An iterator over :attr:`kw_dir` and any of :attr:`ae_dir`,\
                :attr:`af_dir`, :attr:`ve_dir` that are set.
        """
        return iter(
            [
                data_dir
                for data_dir in [
                    self.ae_dir,
                    self.af_dir,
                    self.ve_dir,
                    self.kw_dir,
                ]
                if data_dir
            ],
        )
