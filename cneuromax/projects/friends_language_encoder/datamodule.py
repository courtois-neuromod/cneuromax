"""``project` dataset, datamodule & its config."""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated as An

import pandas as pd
from datasets.arrow_dataset import Dataset as HuggingfaceDataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)

from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)
from cneuromax.utils.beartype import equal, one_of

from .utils import group_texts_hf


def create_dataset(
    csv_file_path: Path,
    tokenizer: PreTrainedTokenizerBase,
) -> HuggingfaceDataset:
    """Creates a :class:`datasets.Dataset` from a .csv file.

    Tokenizes (using :paramref:`tokenizer`) and groups (using
    :func:`~.group_texts_hf`) the raw data into a
    :class:`datasets.Dataset`.

    Args:
        csv_file_path: Path to .csv file for lazy loading.
        tokenizer: See :class:`~.PreTrainedTokenizerBase`.
    """
    data_df = pd.read_csv(csv_file_path, encoding="ISO-8859-1")
    data_df = data_df["line"].to_frame()
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)
    hf_dataset = HuggingfaceDataset.from_pandas(df=data_df)
    hf_tokenized_dataset = hf_dataset.map(
        lambda x: tokenizer(x["line"], return_special_tokens_mask=True),
        batched=True,
        remove_columns=["line"],
    )
    hf_grouped_and_tokenized_dataset = hf_tokenized_dataset.map(
        partial(group_texts_hf, block_size=tokenizer.model_max_length),
        batched=True,
    )
    hf_grouped_and_tokenized_dataset.set_format("torch")
    return hf_grouped_and_tokenized_dataset


@dataclass
class FriendsDataModuleConfig(BaseDataModuleConfig):
    """Holds :class:`FriendsDataModule` config values.

    Args:
        mlm_probability: The probability with which to replace tokens\
            with ``[MASK]`` tokens.
        model_name: The name of the pretrained model.
    """

    mlm_probability: An[float, equal(0.15)] = 0.15
    model_name: str = "${model_name}"


class FriendsDataModule(BaseDataModule):
    """``project`` :class:`.BaseDataModule`.

    Attributes:
        dataset: Tokenize (if annotation) and iterates on the data
    """

    def setup(
        self: "FriendsDataModule",
        stage: An[str, one_of("fit", "validate", "test")],
    ) -> None:
        """See :meth:`~.BaseDataModule.setup`.

        Args:
            stage: See :paramref:`~.BaseDataModule.setup.stage`.
        """
        self.config: FriendsDataModuleConfig
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=self.config.mlm_probability,
        )
        project_data_path = Path(
            f"{self.config.data_dir}/friends_language_encoder/",
        )
        if stage == "fit":
            self.datasets.train = create_dataset(
                csv_file_path=project_data_path / "train.csv",
                tokenizer=tokenizer,
            )
            self.datasets.val = create_dataset(
                csv_file_path=project_data_path / "val.csv",
                tokenizer=tokenizer,
            )
        else:  # stage == "test"
            self.datasets.test = create_dataset(
                csv_file_path=project_data_path / "test.csv",
                tokenizer=tokenizer,
            )
