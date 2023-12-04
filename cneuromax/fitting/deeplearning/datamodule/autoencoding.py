import lightning.pytorch as pl
import numpy as np
import torch
from cntrain.utils.math import normalize
from torch.utils.data import DataLoader, Dataset, random_split


class AEDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.X = torch.tensor(normalize(data), dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index: int):
        return self.X[index]


class AEDataModule(pl.LightningDataModule):
    def __init__(self, data: np.ndarray, batch_size: int):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def setup(self, stage: str):
        dataset = AEDataset(self.data)
        self.train_dataset, self.val_dataset = random_split(
            dataset, [0.9, 0.1]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
