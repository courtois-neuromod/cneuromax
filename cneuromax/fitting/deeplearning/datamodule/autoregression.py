import lightning.pytorch as pl
import numpy as np
import torch
from cntrain.util.math import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


class ARDataset(Dataset):
    def __init__(self, data: list):
        (
            features_strung_array_or_tensor,
            actions_strung_array,
            rew_strung_array,
            done_strung_array,
        ) = data

        # 1) Normalize / PyTorch -> NumPy

        # array: comes from environment -> normalize to curb large values
        if isinstance(features_strung_array_or_tensor, np.ndarray):
            features_strung_array = normalize(features_strung_array_or_tensor)

        # tensor: comes from AE (normalization not needed) -> turn to array
        else:
            features_strung_array = (
                features_strung_array_or_tensor.cpu().numpy()
            )

        # normalize rewards to curb large values
        rew_strung_array = normalize(rew_strung_array)

        # 2) Split strung arrays

        done_idxs = np.where(done_strung_array == True)[0]

        features_split_array_list = np.split(
            features_strung_array, done_idxs + 1
        )[:-1]

        actions_split_array_list = np.split(
            actions_strung_array, done_idxs + 1
        )[:-1]

        rew_split_array_list = np.split(rew_strung_array, done_idxs + 1)[:-1]

        done_split_array_list = np.split(done_strung_array, done_idxs + 1)[:-1]

        # 3) NumPy -> PyTorch

        features_split_tensor_list = [
            torch.from_numpy(array[:-1]) for array in features_split_array_list
        ]
        actions_split_tensor_list = [
            torch.from_numpy(array[:-1]) for array in actions_split_array_list
        ]
        next_features_split_tensor_list = [
            torch.from_numpy(array[1:]) for array in features_split_array_list
        ]
        rew_split_tensor_list = [
            torch.from_numpy(array[1:]) for array in rew_split_array_list
        ]
        done_split_tensor_list = [
            torch.from_numpy(array[1:]) for array in done_split_array_list
        ]

        # 4) Pad sequences

        features_padded_tensor = pad_sequence(
            features_split_tensor_list, batch_first=True
        )
        actions_padded_tensor = pad_sequence(
            actions_split_tensor_list, batch_first=True
        )
        self.features_actions_padded_tensor = torch.cat(
            (features_padded_tensor, actions_padded_tensor), axis=-1
        )

        self.next_features_padded_tensor = pad_sequence(
            next_features_split_tensor_list, batch_first=True
        )
        self.rew_padded_tensor = pad_sequence(
            rew_split_tensor_list, batch_first=True
        )
        self.done_padded_tensor = pad_sequence(
            done_split_tensor_list, batch_first=True
        )

    def __len__(self):
        return self.features_actions_padded_tensor.shape[0] - 1

    def __getitem__(self, index: int):
        return (
            self.features_actions_padded_tensor[index],
            self.next_features_padded_tensor[index],
            self.rew_padded_tensor[index],
            self.done_padded_tensor[index],
        )


class ARDataModule(pl.LightningDataModule):
    def __init__(self, data: list, batch_size: int):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.dataset = ARDataset(self.data)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [0.9, 0.1]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
