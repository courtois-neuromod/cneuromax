import os
import csv
import h5py
import numpy as np
import torch
from tqdm import tqdm
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds
import glob
import pytorch_lightning as pl
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, Dataset


LOAD_CONFOUNDS_PARAMS = {
    "strategy": ["motion", "high_pass", "wm_csf", "global_signal"],
    "motion": "basic",
    "wm_csf": "basic",
    "global_signal": "basic",
    "demean": True,
}


def get_nifti_path_from_events_path(events_path, events_dir, nifti_data_dir):
    """Generate the nifti path corresponding to an events path, handling
    the difference in the zero-padding convention for the run number."""
    run = events_path.split("run-")[1][1]
    nifti_path = (
        events_path.split("run-")[0]
        + "run-"
        + run
        + "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    nifti_path = nifti_path.replace(events_dir, nifti_data_dir)
    return nifti_path


def get_level(bk2_name):
    """Get the string of the level from the name of the bk2 file."""
    if "mario" in bk2_name:
        level = bk2_name.split("level-")[1][:4]
        level = "Level" + level[1] + "-" + level[3]
    elif "shinobi" in bk2_name:
        level = bk2_name.split("level-")[1][0]
        sub_level = "1" if level == "4" else "0"
        level = "Level" + level + "-" + sub_level
    else:
        raise ValueError("game not recognised in bk2 name")
    return level


class HybridDataset(Dataset):
    def __init__(self, load_path, modalities):
        self.load_path = load_path
        self.length = None
        self.load_file = None
        self.modalities = modalities

    def __len__(self):
        if self.length is None:
            with h5py.File(self.load_path, "r") as load_file:
                self.length = max([int(k.split("_")[-1]) for k in load_file.keys()])
        return self.length

    def __getitem__(self, idx):
        # returns samples of shape video:(c, n_frames, h, w) and fMRI:(voxels)
        if self.load_file is None:
            self.load_file = h5py.File(self.load_path, "r")
        item = {}
        for mod in self.modalities:
            item[mod] = torch.from_numpy(self.load_file[f"{mod}_{str(idx)}"][:]).float()
        return item


class HybridDataModule(pl.LightningDataModule):
    """
    Pytorch lightning datamodule with both video and BOLD data.
    Returns batches containing (frames, boold), each one of respsctive shapes
    (bs, n_frames, 3, 64, 64) and (TODO).
    """

    def __init__(
        self,
        lazy_load_dir,
        modalities,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.modalities = modalities
        self.num_workers = num_workers
        self.tng_path = os.path.join(lazy_load_dir, "tng_data.h5")
        self.val_path = os.path.join(lazy_load_dir, "val_data.h5")

    def setup(self, stage):
        pass

    @staticmethod
    def make_lazy_loading_file(
        replay_path,
        rep_list,
        events_dir,
        nifti_data_dir,
        subject,
        lazy_load_path,
        n_frames,  # sequence length
        time_downsample,
        modalities=[],
        original_fps=60,  # before time dowsampling
        t_r=1.49,
        min_lag=3,  # in TRs
        max_lag=5,  # in TRs
        mask_path=None,
        mask_is_atlas=False,
        fwhm=None,
    ):
        if mask_is_atlas:
            masker = NiftiLabelsMasker(
                mask_path, standardize=True, detrend=True, smoothing_fwhm=fwhm
            )
        else:
            masker = NiftiMasker(
                mask_path, standardize=True, detrend=True, smoothing_fwhm=fwhm
            )
        masker.fit()

        events_path_template = os.path.join(
            events_dir, f"{subject}/ses-*/func/*_events.tsv"
        )
        events_paths_list = glob.glob(events_path_template)

        replay_file = h5py.File(replay_path, "r")

        fps = original_fps / time_downsample  # after downsampling
        n_chunks = int(
            (max_lag - min_lag + 1) * t_r * fps / n_frames
        )  # number of frame sequences per bold volume
        n_chunks = max(n_chunks, 1)

        with h5py.File(lazy_load_path, "w") as lazy_load_file:
            index = 0
            for events_path in tqdm(events_paths_list, desc="Events files"):
                nifti_path = get_nifti_path_from_events_path(
                    events_path, events_dir, nifti_data_dir
                )
                if not os.path.isfile(nifti_path):
                    continue
                confounds, _ = load_confounds(nifti_path, **LOAD_CONFOUNDS_PARAMS)
                bold = masker.transform(nifti_path, confounds=confounds)
                with open(events_path, "r") as events_file:
                    events = csv.DictReader(events_file, delimiter="\t")
                    rows = [row for row in events]
                for row in tqdm(rows, desc="bk2 files"):
                    if row["stim_file"] in ["", "Missing file", "n/a"]:
                        continue
                    bk2_name = os.path.splitext(os.path.split(row["stim_file"])[1])[0]
                    level = get_level(bk2_name)
                    if f"{subject}/{level}/{bk2_name}" not in rep_list:
                        continue

                    frames = replay_file[subject][level][bk2_name]["frames"]
                    start = int(float(row["onset"]) / t_r) + max_lag
                    n_samples = len(frames) - len(frames) % time_downsample
                    frames = frames[:n_samples][::time_downsample]
                    frames = frames.astype(np.float32) / 255.0 - 0.5

                    mod_data = {}
                    for mod in modalities:
                        data = replay_file[subject][level][bk2_name][mod][:n_samples]
                        data = sliding_window_view(
                            data, window_shape=time_downsample, axis=0
                        )[::time_downsample]
                        mod_data[mod] = data

                    i = 0
                    while (i * t_r * fps) + n_chunks * n_frames <= len(
                        frames
                    ) and start + i < len(bold):
                        # TODO: check why start + i can be longer than len(bold)
                        mod_chunks = {mod: [] for mod in modalities}
                        frame_chunks = []
                        start_frames = int(i * t_r * fps)
                        for k in range(n_chunks):
                            start_chunk = start_frames + k * n_frames
                            end_chunk = start_frames + (k + 1) * n_frames
                            frame_chunks.append(
                                np.moveaxis(frames[start_chunk:end_chunk], 0, 1)
                            )
                            for mod in modalities:
                                mod_chunks[mod].append(
                                    mod_data[mod][start_chunk:end_chunk]
                                )

                        lazy_load_file.create_dataset(
                            f"frames_{str(index)}",
                            data=np.squeeze(np.stack(frame_chunks, axis=0)),
                        )
                        lazy_load_file.create_dataset(
                            f"bold_{str(index)}", data=bold[start + i]
                        )
                        for mod in modalities:
                            lazy_load_file.create_dataset(
                                f"{mod}_{str(index)}",
                                data=np.squeeze(np.stack(mod_chunks[mod], axis=0)),
                            )
                        index += 1
                        i += 1
                    if start + i >= len(bold):
                        print("===========MORE FRAMES THAN BOLD==================")
                        print(nifti_path)
                        print(
                            bk2_name,
                            "onset",
                            row["onset"],
                            "start",
                            start,
                            "i",
                            i,
                            "len(bold)",
                            len(bold),
                            "len(frames)",
                            len(frames),
                        )
                        print(
                            "frames_duration",
                            len(frames) / fps,
                            "onset + frames duration",
                            float(row["onset"]) + len(frames) / fps,
                            "bold duration",
                            len(bold) * t_r,
                        )
                        print("==================================================")

    def train_dataloader(self):
        tng_dataset = HybridDataset(self.tng_path, self.modalities)
        return DataLoader(
            tng_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dataset = HybridDataset(self.val_path, self.modalities)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
