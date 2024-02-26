""":class:`UncKWPredDataset` & its config."""

from dataclasses import dataclass
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from .utils import KWPredDatasetPaths, create_load_function


@dataclass
class UncKWPredDatasetConfig:
    """:class:`UncKWPredDataset` config.

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
        ├─ ISD-Sust-006-Engine.csv
        └─ MID2KLK_xref.csv


    Args:
        root_data_dir: Path to the high-level data directory.
        klk_wavs_rel_dir: Relative path with respect to\
            :paramref:`root_data_dir` to the directory containing\
            ``.wav`` files extracted from ``.klk`` files. See\
            :mod:`.kw_pred` for more details on ``.klk`` ``.wav``\
            predictions (KW).
        int_rel_file: Relative path with respect to\
            :paramref:`root_data_dir` to the file containing the\
            intentions.
        conv_rel_file: Relative path with respect to\
            :paramref:`root_data_dir` to the file containing the\
            conversions from Movie IDs to KLK IDs.
    """

    root_data_dir: str = "/media/DATA/"
    klk_wavs_rel_dir: str = "HEMC_klk_wavs/"
    int_rel_file: str = "ISD-Sust-006-Engine.csv"
    conv_rel_file: str = "MID2KLK_xref.csv"


class UncKWPredDataset(Dataset[dict[str, Tensor]]):
    """:mod:`.unc_kw_pred` :class:`torch.utils.data.Dataset`.

    Args:
        config: See :class:`KWPredDataConfig`.

    Attributes:
        config (:class:`KWPredDataConfig`): See :paramref:`config`.

        content_ids (``list[int]``): List of valid content IDs.
    """

    def __init__(
        self: "UncKWPredDataset",
        config: UncKWPredDatasetConfig,
    ) -> None:
        self.config = config
        paths = UncKWPredDatasetPaths(
            klk_wavs_dir=Path(config.root_data_dir + config.klk_wavs_rel_dir),
        )
        self.load_data, self.num_data_points = create_load_function(
            paths=paths,
        )

    def __len__(self: "UncKWPredDataset") -> int:
        """See :meth:`torch.utils.data.Dataset.__len__`."""
        return len(self.num_data_points)

    def __getitem__(self: "UncKWPredDataset", idx: int) -> dict[str, Tensor]:
        """See :meth:`torch.utils.data.Dataset.__getitem__`."""
        while True:  # spooky (~'o')~ ...
            try:
                return self.load_data(idx=idx)
            except Exception:  # noqa: PERF203, BLE001
                idx = (idx + 1) % len(self.num_data_points)
