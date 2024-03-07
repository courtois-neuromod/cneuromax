""":class:`ReplayDataModule`."""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset
from tqdm import tqdm

from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)


def uniform_dowsample_frames(
    frames: np.ndarray,
    n_frames: int,
    time_downsample: int,
    stride: int | None = None,
) -> np.ndarray:
    """Dowsample the frames.

    Taking one out of every `time_downsample` frames, effectively
    dividing the original framerate by `time_downsample`.
    Then make chunks of `n_frames` downsampled frames.

    Args:
        frames: np.Array, raw frames
        n_frames: int, number of frames returned per chunk
        time_downsample: int, by how much to divide the framerate
        stride: int, number of downsampled frames between the 1st frame
            of two consecutive chunks. If None it will be set to
            n_frames. Defaults to None.

    Returns:
        frame_chuks: np.Array of shape (n_chunks, 3, n_frames, H, W)
    """
    stride = stride if stride is not None else n_frames
    n_raw_frames = len(frames) - len(frames) % time_downsample
    frames = frames[:n_raw_frames:time_downsample]
    frame_chunks = sliding_window_view(frames, window_shape=n_frames, axis=0)[
        ::stride
    ]
    return np.moveaxis(frame_chunks, -1, 2)


def uniform_dowsample_modality(
    data: np.ndarray,
    n_frames: int,
    time_downsample: int,
    stride: int | None = None,
) -> np.ndarray:
    """Dowsample the other modalities.

    Similar as for frames, but keep the values of the removed frames on
    an extra dimension.
    E.g. if n_frames=2, time_downsample=3, stride=None:
    raw frames     [0,  1,  2,  3,  4,  5,   6,  7,  8,  9,  10,  11]
    frame chunks  [[0,          3],         [6,          9]]
    modality     [[[0,  1,  2],[3,  4,  5]],[[6, 7,  8],[9,  10,  11]]]
    Technically it's not really downsampling but reshaping.

    Args:
        data: np.Array
        n_frames: int, number of frames per chunk
        time_downsample: int, by how much to divide the framerate
        stride: int, number of downsampled frames between the 1st frame
            of two consecutive chunks. If None it will be set to
            n_frames. Defaults to None.

    Returns:
        data: np.Array
    """
    stride = stride if stride is not None else n_frames
    n_raw_frames = len(data) - len(data) % time_downsample
    data = data[:n_raw_frames]
    data = sliding_window_view(data, window_shape=time_downsample, axis=0)[
        ::time_downsample
    ]
    data = sliding_window_view(data, window_shape=n_frames, axis=0)[::stride]
    return np.moveaxis(data, 2, 1)


class ReplayDataset(Dataset):
    """Replay Dataset.

    Args:
        load_path: str, path to the hdf5 file for lazy loading
        modalities: list of str, modalities to return
    """

    def __init__(
        self: "ReplayDataset",
        load_path: str,
        modalities: list,
    ) -> None:
        self.load_path = load_path
        with h5py.File(self.load_path, "r") as load_file:
            self.length = max([int(s.split("_")[1]) for s in load_file]) + 1
        self.load_file = None
        self.modalities = modalities

    def __len__(self: "ReplayDataset") -> int:
        """Returns the length of the dataset."""
        return self.length

    def __getitem__(self: "ReplayDataset", idx: int) -> dict:
        """Get item.

        Returns samples of shape video:(c, n_frames, h, w) and
        other:(time_dowsample).
        """
        if self.load_file is None:
            self.load_file = h5py.File(self.load_path, "r")
        item = {}
        for mod in self.modalities:
            item[mod] = torch.from_numpy(
                self.load_file[f"{mod}_{idx}"][:],
            ).float()
        return item


@dataclass
class ReplayDataModuleConfig(BaseDataModuleConfig):
    """Holds :class:`ReplayDataModule` config values.

    Args:
        data_path: str, path to the hdf5 file with frames
        lazy_load_dir: str, path to the directory where to save the lazy
            loaded files
        split_file: str, path to the json file containing the training
            and validation split of the runs.
        subject: str, subject id, or 'all' for all subjects
        n_frames: int, number of frames in one sample
        modalities: list of str, name of the modalities to use (other
            than frames), only frames are used if the list is empty.
        time_downsample: int, Factor by which the frame frequency is
            divided, i.e. one out of every `time_downsample` frames will
            be kept. Defaults to 5.
        stride: int, stride between samples, None will make it equal to
            n_frames. Defaults to None.
        fps: int, Numbre of frames per seconds in the original data.
            Defaults to 60.
    """

    data_path: str = "data/mario_frames_128.h5"
    lazy_load_dir: str = "data/lazy_load"
    split_file: str = "data/basic_split.json"
    subject: str = "all"
    n_frames: int = 16
    modalities: list[str] | None = None
    time_downsample: int = 5
    stride: int = None
    fps: int = 60


class ReplayDataModule(BaseDataModule):
    """``project`` :class: `pl.LightningDataModule`.

    Returns batches containing (input, output),
    each one of shape (bs, n_in/n_out, 3, 64, 64).

    Args:
        lazy_load_dir: path of the directory containing the training
            and validation hdf5 lazy loading files
        modalities: list of str, modalities to use in addition to
            'frames'
        batch_size: int, batch size
        num_workers: int, number of workers
    """

    def __init__(
        self: "ReplayDataModule",
        config: ReplayDataModuleConfig,
    ) -> None:
        super().__init__(config=config)
        self.tng_path = Path(config.lazy_load_dir) / "tng_data.h5"
        self.val_path = Path(config.lazy_load_dir) / "val_data.h5"

    def setup(self: "ReplayDataModule", stage: str) -> None:
        """Create the lazy loading files and the datasets."""
        modalities = self.cfg.modalities or []
        if stage == "fit":
            Path(self.cfg.lazy_load_dir).mkdir(parents=True, exist_ok=True)
            with Path(self.cfg.split_file).open() as split_file:
                run_split = json.load(split_file)
            tng_list = run_split["training"]
            val_list = run_split["validation"]
            if self.cfg.subject != "all":
                tng_list = [p for p in tng_list if self.cfg.subject in p]
                val_list = [p for p in val_list if self.cfg.subject in p]
            self.make_lazy_loading_file(
                self.config.data_path,
                self.tng_path,
                tng_list,
                self.cfg.n_frames,
                modalities,
                self.cfg.time_downsample,
            )
            self.make_lazy_loading_file(
                self.config.data_path,
                self.val_path,
                val_list,
                self.cfg.n_frames,
                modalities,
                self.cfg.time_downsample,
            )
            self.datasets.train = ReplayDataset(
                self.tng_path,
                ["frames", *self.modalities],
            )
            self.datasets.val = ReplayDataset(
                self.val_path,
                ["frames", *self.modalities],
            )

    @staticmethod
    def make_lazy_loading_file(  # noqa: PLR0913
        data_path: str,
        lazy_load_path: str,
        data_list: list[str],
        n_frames: int,
        modalities: list[str],
        time_downsample: int,
        stride: int | None = None,
        fps: int = 60,
    ) -> None:
        """Generate a HDF5 file for lazy loading.

        The frames are uniformly downsampled, effectively dividing the
        original framerate by `time_downsample`.

        Args:
            data_path: str, path to the HDF5 file containing the frames
                and modalities lazy_load_path: str, path to the HDF5 to
                create
            lazy_load_path: str, path to the HDF5 file to create for
                lazy loading
            data_list: list of str, list of the HDF5 dataset paths in
                the data file to use
            n_frames: int, number of frames in one sample
            modalities: list of str, name of the other modalities to
                use, only frames are used if the list is empty
            time_downsample: int, Factor by which the frame frequency is
                divided, i.e. one out of every `time_downsample` frames
                will be kept
            stride: int, stride between samples, None will make it equal
                to n_frames. Defaults to None.
            fps: int, Numbre of frames per seconds in the original data.
                Defaults to 60.
        """
        index = 0
        data_file = h5py.File(data_path, "r")
        with h5py.File(lazy_load_path, "w") as lazy_load_file:
            for data_h5_path in tqdm(
                data_list,
                desc="Creating lazy load file",
            ):
                frames = data_file[data_h5_path]["frames"][:]
                if len(frames) < 30 * fps:  # more than 30s of gameplay
                    continue
                frame_chunks = uniform_dowsample_frames(
                    frames,
                    n_frames,
                    time_downsample,
                    stride,
                )
                if frame_chunks.max() > 1:
                    frame_chunks = (
                        frame_chunks.astype(np.float32) / 255.0 - 0.5
                    )

                mod_data = {}
                for mod in modalities:
                    data = data_file[data_h5_path][mod][:]
                    mod_data[mod] = uniform_dowsample_modality(
                        data,
                        n_frames,
                        time_downsample,
                        stride,
                    )

                for i, chunk in enumerate(frame_chunks):
                    lazy_load_file.create_dataset(
                        f"frames_{index}",
                        data=chunk,
                    )
                    for mod in modalities:
                        lazy_load_file.create_dataset(
                            f"{mod}_{index}",
                            data=mod_data[mod][i],
                        )
                    index += 1
