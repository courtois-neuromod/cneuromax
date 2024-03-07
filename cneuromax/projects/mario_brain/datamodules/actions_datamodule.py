import os
import h5py
from tqdm import tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# Only keep used buttons : A, up, down, left, right, B
USED_BUTTONS = [True, False, False, False, True, True, True, True, True]


class ActionsDataset(Dataset):
    def __init__(
        self, load_path, n_frames, downsample, n_actions_context=0, fast_training=False
    ):
        """
        Args:
            load_path: str, path to the hdf5 file for lazy loading
            n_frames: int, number of frames per sample
            downsample: temporal downsampling for the frames
            n_actions_context: int, number of action vectors to return in addition
                to the target action (the one right after the last input frame). These
                actions are the ones just before the target action. Defaults to 0.
            fast_training: bool, wether to use fats training
                (providing next sequence of frames) or not, defaults to False
        """
        self.load_path = load_path
        self.n_frames = n_frames
        self.n_actions_context = n_actions_context
        self.downsample = downsample
        self.h5_groups = []
        self.idx_shift = {}
        shift = 0
        self.fast_training = fast_training
        factor = 2 if fast_training else 1
        with h5py.File(self.load_path, "r") as load_file:
            h5_group_names = sorted([key for key in load_file.keys()])
            for name in h5_group_names:
                self.idx_shift[name] = shift
                n_samples = load_file[name]["frames"].len() - (
                    (factor * n_frames - 1) * downsample + 1
                )
                self.h5_groups += [name] * n_samples
                shift += n_samples

        self.length = shift
        self.load_file = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.load_file is None:
            self.load_file = h5py.File(self.load_path, "r")

        h5_group = self.h5_groups[idx]
        shift = self.idx_shift[h5_group]
        start_frame = idx - shift
        target_frame = idx - shift + self.n_frames * self.downsample
        if self.fast_training:
            end_frame = target_frame + self.n_frames * self.downsample
        else:
            end_frame = target_frame
        frames = self.load_file[h5_group]["frames"][
            start_frame : end_frame : self.downsample
        ]
        frames = frames.astype(np.float32) / 255.0 - 0.5
        target_action_id = target_frame - self.downsample + 1
        target_actions = self.load_file[h5_group]["actions"][target_action_id]
        target_actions = target_actions[USED_BUTTONS]
        item = {
            "frames": torch.from_numpy(frames).movedim(0, 1).float(),
            "target_actions": torch.from_numpy(target_actions).float(),
        }
        if self.n_actions_context:
            context_actions = self.load_file[h5_group]["actions"][
                target_action_id - self.n_actions_context : target_action_id
            ]
            context_actions = context_actions[:, USED_BUTTONS]
            item["context_actions"] = context_actions
        return item


class ActionsDataModule(pl.LightningDataModule):
    """
    Pytorch lightning datamodule for predicting actions from frames.
    """

    def __init__(
        self,
        lazy_load_dir,
        batch_size,
        num_workers,
        n_frames,
        downsample,
        n_actions_context=0,
        fast_training=False,
    ):
        """_
        Args:
            lazy_load_dir: path of the directory containing the training
                and validation hdf5 lazy loading files
            batch_size: int, batch size
            num_workers: int, number of workers
            n_frames: int, number of frames per sample
            downsample: int, temporal downsampling for the frames
            n_actions_context: int, number of actions preceding the one to predict
                to return
            fast_training: bool, wether to use fast training
                (providing next sequence of frames) or not, defaults to False
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_frames = n_frames
        self.downsample = downsample
        self.tng_path = os.path.join(lazy_load_dir, "tng_data.h5")
        self.val_path = os.path.join(lazy_load_dir, "val_data.h5")
        self.fast_training = fast_training
        self.n_actions_context = n_actions_context

    def setup(self, stage):
        pass

    @staticmethod
    def make_lazy_loading_file(
        data_path,
        lazy_load_path,
        data_list,
    ):
        """Generate a HDF5 file to be used by a dataloader for lazy loading.

        Args:
            data_path: str, path to the HDF5 file containing the frames and modalities
            lazy_load_path: str, path to the HDF5 to create
            data_list: list of str, list of the HDF5 dataset paths in the data file to
                use.
        """
        data_file = h5py.File(data_path, "r")
        with h5py.File(lazy_load_path, "w") as lazy_load_file:
            for data_h5_path in tqdm(data_list, desc="Creating lazy load file"):
                frames = data_file[data_h5_path]["frames"][:]
                actions = data_file[data_h5_path]["actions"][:]
                # if len(frames) < 30 * 60:  # more than 30s of gameplay
                #    continue
                group = lazy_load_file.create_group(data_h5_path.split("/")[-1])
                group.create_dataset("frames", data=frames)
                group.create_dataset("actions", data=actions)

    def train_dataloader(self):
        tng_dataset = ActionsDataset(
            self.tng_path,
            self.n_frames,
            self.downsample,
            self.n_actions_context,
            self.fast_training,
        )
        return DataLoader(
            tng_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        val_dataset = ActionsDataset(
            self.val_path,
            self.n_frames,
            self.downsample,
            self.n_actions_context,
            self.fast_training,
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
