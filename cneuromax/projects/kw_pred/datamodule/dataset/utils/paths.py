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
        an_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.annot_rel_dir`\
            or :obj:`None`.
        kw_dir: Concatenation of\
            :paramref:`~.KWPredDatasetConfig.root_data_dir` and\
            :paramref:`~.KWPredDatasetConfig.klk_wavs_rel_dir`.
    """

    ae_dir: Path | None
    af_dir: Path | None
    ve_dir: Path | None
    an_dir: Path | None
    kw_dir: Path

    def __post_init__(self: "KWPredDatasetPaths") -> None:
        """Post-initialization checks.

        If :attr:`an_dir` is set, the task is unconditional generation.
        If :attr:`ae_dir`, :attr:`af_dir` and/or :attr:`ve_dir` are set,
        the task is conditional generation.

        Raises:
            ValueError: If :attr:`an_dir` is set and any of\
                :attr:`ae_dir`, :attr:`af_dir` and/or :attr:`ve_dir`\
                are also set.
        """
        if self.an_dir and (self.ae_dir or self.af_dir or self.ve_dir):
            error_msg = (
                "Annotations are only meant to be used in an unconditional "
                "generation setting. Make sure to set `annot_rel_dir` to "
                "`None`."
            )
            raise ValueError(error_msg)

    def __iter__(self: "KWPredDatasetPaths") -> Iterator[Path | None]:
        """See return.

        Returns:
            If :attr:`an_dir` is set, an iterator over\
                :attr:`an_dir` and :attr:`kw_dir`. If not, an iterator\
                over :attr:`kw_dir` and any of :attr:`ae_dir`,\
                :attr:`af_dir`, :attr:`ve_dir` that are set.
        """
        if self.an_dir:
            return iter([self.an_dir, self.kw_dir])
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
