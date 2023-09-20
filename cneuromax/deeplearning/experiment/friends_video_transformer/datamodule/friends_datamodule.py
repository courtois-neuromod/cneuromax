"""Adapted from examplar MNIST DataModule."""
import glob
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from omegaconf import DictConfig, OmegaConf
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm

from cneuromax.common.utils.annotations import (
    non_empty_str,
    str_is_fit_or_test,
)
from cneuromax.deeplearning.common.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)


@dataclass
class FriendsSubDataset(Dataset):
    """.

    Attributes:
        load_path: Path to .hdf5 file for lazy loading.
        modalities: modalities to return, e.g. ['frames', 'bold'].
        load_file: loaded .hdf5 file to get items from.
        length: number of items available.
    """

    load_path: str
    modalities: list[str]
    load_file: h5py.File = field(init=False, repr=False)
    length: int = field(init=False)

    def __post_init__(self: "FriendsSubDataset") -> None:
        """Initializes additional attributes."""
        self.load_file: h5py.File = h5py.File(self.load_path, "r")
        self.length: int = (
            max([int(key.split("_")[1]) for key in self.load_file]) + 1
        )

    def __len__(self: "FriendsSubDataset") -> int:
        """Returns number of items in load_file."""
        return self.length

    def __getitem__(self: "FriendsSubDataset", idx: int) -> dict:
        """.

        Returns:
            The indexed item's data per modality inside a dictionary.
            Shape of video frames data: (c, n_frames, h, w)
        """
        item = {}
        for mod in self.modalities:
            item[mod] = torch.from_numpy(
                self.load_file[f"{mod}_{idx}"][:],
            ).float()
        return item


@dataclass
class FriendsDataset:
    """.

    Attributes:
        train: .
        val: .
        test: .
        predict: .
    """

    train: FriendsSubDataset | None = None
    val: FriendsSubDataset | None = None
    test: FriendsSubDataset | None = None
    predict: FriendsSubDataset | None = None


@dataclass
class FriendsDataModuleConfig(BaseDataModuleConfig):
    """.

    Attributes:
        subject: subject number, e.g., sub-01.
        episode_list: list of .
        per_device_num_workers: Per-device number of CPU processes to
            use for data loading (``0`` means that the data will be
            loaded by each device's assigned CPU process)
        device_type: The compute device type to use.
    """

    subject: str | None
    episode_list: DictConfig
    use_bold: bool
    modalities: list[str]

    mkv_data_dir: non_empty_str
    lazy_loading_dir: non_empty_str

    nifti_data_dir: str | None
    load_confounds_params: DictConfig | None

    time_downsample: int
    tr: float = 1.49  # TR length in seconds
    original_fps: float = 29.97  # frames per second; float = 60.0
    n_chunks: int = 3  # number of frame sequences per bold volume (TR)
    resize_dim: int = 128
    dtype: str = "float32"
    min_lag: int = 3  # in TRs
    max_lag: int = 5  # in TRs

    mask_path: str | None = None
    mask_is_atlas: bool = False
    fwhm: int | None = None


class FriendsDataModule(BaseDataModule):
    """.

    Attributes:
        config : subject-specific config parameters.
        dataset (``transforms.Compose``): The ``torchvision`` dataset
            transformations.
    """

    def __init__(
        self: "FriendsDataModule",
        config: FriendsDataModuleConfig,
    ) -> None:
        """.

        Calls parent constructor, type-hints the config, sets the
        train/val split and creates the dataset transform.

        Args:
            config: .
        """
        super().__init__(config)
        self.config: FriendsDataModuleConfig = config
        self.dataset: FriendsDataset = FriendsDataset()

    @staticmethod
    def make_lazy_loading_file(
        mkv_data_dir: str,
        *,
        nifti_data_dir: str | None,
        lazy_load_path: str,
        subject: str | None,
        episode_list: list[str],
        time_downsample: int,
        use_bold: bool = False,
        load_confounds_params: DictConfig | None = None,
        original_fps: float = 29.97,  # float = 60.0
        resize_dim: int = 128,
        dtype: str = "float32",
        tr: float = 1.49,
        n_chunks: int = 3,  # number of frame sequences per bold volume
        max_lag: int = 5,  # in TRs
        mask_path: str | None = None,
        mask_is_atlas: bool = False,
        fwhm: int | None = None,
    ) -> None:
        """.

        Creates .h5 file from .mkv files to load batches of frames (and
        other modalities, like bold data) locally from slurm.
        """
        if use_bold:
            if mask_is_atlas:
                masker = NiftiLabelsMasker(
                    mask_path,
                    standardize=True,
                    detrend=True,
                    smoothing_fwhm=fwhm,
                )
            else:
                masker = NiftiMasker(
                    mask_path,
                    standardize=True,
                    detrend=True,
                    smoothing_fwhm=fwhm,
                )
            masker.fit()

        lazy_load_file: h5py.File = h5py.File(lazy_load_path, "w")
        n_frames: int = int(tr * original_fps)
        # fps after downsampling
        # fps: float = original_fps / time_downsample

        index: int = 0
        for episode in tqdm(episode_list, desc=".mkv files"):
            mkv_path = f"{mkv_data_dir}/s{episode[2]}/friends_{episode}.mkv"
            if Path(mkv_path).exists():
                if use_bold:
                    nii_list: list[str] = glob.glob(
                        f"{nifti_data_dir}/{subject}/ses-0*/func/"
                        f"{subject}_ses-0*_task-{episode}_space-T1w_desc-preproc_bold.nii.gz",
                    )

                    if len(nii_list) != 1 or not Path(nii_list[0]).exists():
                        continue

                    confounds, _ = load_confounds(
                        nii_list[0],
                        **OmegaConf.to_container(load_confounds_params),
                    )
                    bold = masker.transform(nii_list[0], confounds=confounds)

                chunk_dict: dict = {}

                cap: cv2.VideoCapture = cv2.VideoCapture(mkv_path)

                frame_count: int = 0

                success: bool = True
                image: np.array | None = None

                while success:
                    # while success and len(chunk_dict) < 50:
                    chunk_frames: list[np.array] = []

                    while frame_count < int(
                        (len(chunk_dict) + 1) * tr * original_fps
                    ):
                        success, image = cap.read()

                        if success:
                            # flip color channels to be RGB (from cv2)
                            chunk_frames.append(
                                np.floor(
                                    resize(
                                        image[..., ::-1],
                                        (resize_dim, resize_dim),
                                        preserve_range=True,
                                        anti_aliasing=True,
                                    ),
                                ).astype("uint8"),
                            )
                            frame_count += 1

                    # only process complete chunks
                    if success:
                        # temporal downsampling
                        frames: np.array = (
                            np.asarray(chunk_frames[:n_frames], dtype=dtype)[
                                ::time_downsample
                            ]
                            / 255.0
                            - 0.5
                        )

                        """pytorch / chainer input dimension order:
                        Channel x Frame x Height x Width
                        F, H, W, C -> C, F, H, W
                        """
                        chunk_array: np.array = np.transpose(
                            frames,
                            [3, 0, 1, 2],
                        )
                        # must be color channel
                        if not chunk_array.shape[0] == 3:
                            raise ValueError

                        chunk_dict[len(chunk_dict)] = chunk_array

                cap.release()

                start: int = max_lag
                i: int = 0

                length_bold: int = 2000 if not use_bold else len(bold)

                while (i + (n_chunks - 1)) < len(
                    chunk_dict,
                ) and start + i < length_bold:
                    frame_chunks: list[np.array] = [
                        chunk_dict[i + k] for k in range(n_chunks)
                    ]
                    lazy_load_file.create_dataset(
                        f"frames_{index}",
                        data=np.squeeze(np.stack(frame_chunks, axis=0)),
                    )
                    if use_bold:
                        lazy_load_file.create_dataset(
                            f"bold_{index}",
                            data=bold[start + i],
                        )

                    index += 1
                    i += 1

        lazy_load_file.close()

    def prepare_data(
        self: "FriendsDataModule",
        lazy_load_path: str,
        set_name: str,
    ) -> None:
        """Builds the .hdf5 to load as subdataset."""
        c: FriendsDataModuleConfig = self.config

        FriendsDataModule.make_lazy_loading_file(
            mkv_data_dir=c.mkv_data_dir,
            nifti_data_dir=c.nifti_data_dir,
            lazy_load_path=lazy_load_path,
            subject=c.subject,
            episode_list=c.episode_list[set_name],
            time_downsample=c.time_downsample,
            use_bold=c.use_bold,
            load_confounds_params=c.load_confounds_params,
            original_fps=c.original_fps,  # float = 60.0
            resize_dim=c.resize_dim,
            dtype=c.dtype,
            tr=c.tr,
            n_chunks=c.n_chunks,  # num of frame sequences per bold volume
            max_lag=c.max_lag,  # in TRs
            mask_path=c.mask_path,
            mask_is_atlas=c.mask_is_atlas,
            fwhm=c.fwhm,
        )

    def setup(
        self: "FriendsDataModule",
        stage: str_is_fit_or_test,
    ) -> None:
        """Creates the train/val/test datasets.

        Args:
            stage: 'fit' or 'test'.
        """
        ll_dir = self.config.lazy_loading_dir
        set_list: list[str] = ["train", "val"] if stage == "fit" else ["test"]

        for set_name in set_list:
            if self.config.use_bold:
                ll_path: str = f"{ll_dir}/{self.config.subject}_{set_name}.h5"
            else:
                ll_path: str = f"{ll_dir}/{set_name}.h5"

            if not Path(ll_path).exists():
                self.prepare_data(lazy_load_path=ll_path, set_name=set_name)

            setattr(
                self.dataset,
                set_name,
                FriendsSubDataset(
                    load_path=ll_path,
                    modalities=self.config.modalities,
                ),
            )
