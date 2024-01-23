"""Adapted from examplar MNIST DataModule."""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated as An

import pandas as pd
import torch
from datasets import DatasetDict
from datasets.arrow_dataset import Dataset as HuggingfaceDataset
from omegaconf import MISSING
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)

from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)
from cneuromax.utils.beartype import equal

from .friends_util import (
    group_texts,
    tokenize_function,
)


class FriendsDataset(Dataset):
    """.

    Attributes:
        data_dir: Path to .hdf5 file for lazy loading.
        mod: modalities to return, e.g. ['annotations', 'bold'].
        subject: the subject id referring to the subject folder
        load_file: loaded .hdf5 file to get bold data from.
        length: number of items available.
    """

    def __init__(
        self: "FriendsDataset",
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        """Loads data and find the data length.

        Args:
            data_dir: Path to .hdf5 file for lazy loading.
            tokenizer: Tokenizer to use.
            max_seq_length: maximum length of the sequence
            chunk_size: the size of the chunks of text which is
                equal or less than max_seq_length
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        if chunk_size:
            self.chunk_size = chunk_size
        else:
            self.chunk_size = max_seq_length

        self.data = pd.read_csv(self.data_dir, encoding="ISO-8859-1")
        self.data = self.data.drop(columns=["season", "episode"])
        self.data = self.data["line"].to_frame()
        self.data = self.data.dropna()  # remove nan values
        self.data = self.data.reset_index(
            drop=True,
        )

        self.dataset = DatasetDict()
        self.dataset["data"] = HuggingfaceDataset.from_pandas(self.data)

        if self.max_seq_length is None:
            self.max_seq_length = tokenizer.model_max_length
        else:
            self.max_seq_length = min(
                self.max_seq_length,
                tokenizer.model_max_length,
            )

        tokenized_datasets = self.dataset.map(
            partial(tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["line"],
        )

        self.tokenized_data = tokenized_datasets.map(
            partial(group_texts, chunk_size=self.chunk_size),
            batched=True,
        )

        self.length = len(self.tokenized_data)

    def __len__(self: "FriendsDataset") -> int:
        """.

        Returns number of items in load_file.
        """
        return self.length

    def __getitem__(
        self: "FriendsDataset",
        index: int,
    ) -> dict[int, torch.Tensor]:
        """.

        Returns:
            The indexed item's data per modality inside a dictionary.

        TODO: Describe what comes out of this bad boy.

        """
        return self.tokenized_data[index]


@dataclass
class FriendsDataModuleConfig(BaseDataModuleConfig):
    """.

    Attributes:
        data_dir: .
        tokenizer_name:.
        mlm_probability: .

    """

    tokenizer_name: str = MISSING
    mlm_probability: An[float, equal(0.15)] = 0.15


class FriendsDataModule(BaseDataModule):
    """.

    Attributes:
        config : config parameters
        dataset: Tokenize (if annotation) and iterates on the data
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
            tokenizer: Tokenizer to use.
        """
        super().__init__(config)
        self.config: FriendsDataModuleConfig = config
        self.tokenizer_name = self.config.tokenizer_name
        self.tokenizer_dir = str(
            Path(self.config.data_dir)
            / "friends_language_encoder"
            / "models"
            / self.tokenizer_name,
        )

    def setup(
        self: "FriendsDataModule",
        stage: str | None = None,
    ) -> None:
        """Creates the train/val/test datasets.

        Args:
            stage: 'fit' or 'test'.
        """
        # Load the dataset using the provided dataset module
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            config=AutoConfig.from_pretrained(self.tokenizer_name),
        )

        max_seq_length = tokenizer.model_max_length

        stages = ["train", "val"] if stage == "fit" else ["test"]

        datasets = {}

        for data_stage in stages:
            dataset = FriendsDataset(
                str(
                    Path(self.config.data_dir)
                    / "friends_language_encoder"
                    / data_stage
                    / f"{data_stage}_set.csv",
                ),
                tokenizer,
                max_seq_length,
            )
            datasets[data_stage] = dataset

        if stage == "fit":
            self.dataset.train = datasets["train"]
            self.dataset.val = datasets["val"]
        else:
            self.dataset.test = datasets["test"]

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=self.config.mlm_probability,
        )
