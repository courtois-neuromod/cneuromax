""":class:`KLKWavDataset`."""
from pathlib import Path

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class KLKWavDataset(Dataset[dict[str, Tensor]]):
    """:mod:`klk_wav`` :class:`torch.utils.data.Dataset`.

    The dataset file structure is of the form:

    .. code-block:: text

        HEMC_klk_wavs/
        ├── ID3026/
        │   ├── ID3026_BL.wav
        │   ├── ID3026_BR.wav
        │   ├── ID3026_FL.wav
        │   └── ID3026_FR.wav
        ├── ID3027/
        │   ├── ID3027_BL.wav
        │   └── ...
        └── ...

    Args:
        data_dir: See :paramref:`~.BaseSubtaskConfig.data_dir`.
    """

    def __init__(self: "KLKWavDataset", data_dir: str) -> None:
        dataset_dir = Path(f"{data_dir}/HEMC_klk_wavs/")
        # Get all folders in dataset_dir
        dataset_dir_entries = dataset_dir.iterdir()
        # Include only folders of the form IDXXXX, where XXXX is a
        # 4-digit number
        id_folders = [
            f
            for f in dataset_dir_entries
            if f.is_dir()
            and f.name.startswith("ID")
            and len(f.name) == 6  # noqa: PLR2004
            and f.name[2:].isdigit()
        ]
        self.dataset: dict[str, dict[str, Tensor]] = {}
        self.dataset_XXXX_list: list[str] = []
        for id_folder in id_folders:
            # Get all files in folder
            id_folder_entries = id_folder.iterdir()
            # Include only files of the form IDXXXX_YY.wav, where XXXX
            # is a 4-digit number and YY is one of BL, BR, FL, FR
            id_folder_files = [
                f
                for f in id_folder_entries
                if f.is_file()
                and f.name.startswith("ID")
                and len(f.name) == 13  # noqa: PLR2004
                and f.name[2:6].isdigit()
                and f.name[7:9] in ["BL", "BR", "FL", "FR"]
                and f.name.endswith(".wav")
            ]
            # Add the files to a nested dictionary, where the first
            # keys are the XXXX digits and the second keys are the
            # YY strings
            self.dataset[id_folder.name[2:6]] = {}
            for dataset_file in id_folder_files:
                wave_tensor, _ = torchaudio.load(dataset_file)
                wave_tensor = wave_tensor.squeeze()
                self.dataset[id_folder.name[2:6]][
                    dataset_file.name[7:9]
                ] = wave_tensor
            # Make sure each YY is represented
            for yy in ["BL", "BR", "FL", "FR"]:
                if yy not in self.dataset[id_folder.name[2:6]]:
                    error_msg = f"Missing {yy} file in {id_folder.name}"
                    raise ValueError(error_msg)
            self.dataset_XXXX_list.append(id_folder.name[2:6])

    def __len__(self: "KLKWavDataset") -> int:
        """See :meth:`torch.utils.data.Dataset.__len__`."""
        return len(self.dataset_XXXX_list)

    def __getitem__(self: "KLKWavDataset", idx: int) -> dict[str, Tensor]:
        """See :meth:`torch.utils.data.Dataset.__getitem__`."""
        return self.dataset[self.dataset_XXXX_list[idx]]
