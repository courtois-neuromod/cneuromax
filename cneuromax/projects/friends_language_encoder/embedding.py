"""."""

import os
import pickle
import string
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from peft.tuners.lora.config import LoraConfig
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_constant_schedule,
)

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModuleConfig,
)
from cneuromax.projects.friends_language_encoder.litmodule_peft import (
    FriendsFinetuningModel,
)
from cneuromax.projects.friends_language_encoder.utils import (
    list_episodes,
    list_seasons,
    preprocess_words,
    set_output,
)


@dataclass
class DataConfigBase:
    """."""

    sweep: str = "overrides#lr~0.0001161#wd~0.001303"
    tsv_path: str = "./data/friends_language_encoder/stimuli/"
    stimuli_dir: str = "./data/friends_language_encoder/stimuli/gpt2"
    base_model_name: str = "gpt2"
    task_type = "CAUSAL_LM"
    inference_mode = False
    r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    fan_in_fan_out = True
    context_size = 50
    bsz = 32
    feature_count = 768
    num_hidden_layers = 12


data_config = DataConfigBase()


class PrepareTokenizedTextBatches:
    """."""

    def __init__(
        self: "PrepareTokenizedTextBatches",
        tokenizer: PreTrainedTokenizerBase,
        tsv_path: str,
        season: str,
        episode: str,
        context_size: int,
        connection_character: str = "Ġ",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = tokenizer.model_max_length
        self.tsv_path = os.path.join(
            tsv_path,
            f"{season}",
            f"friends_{episode}.tsv",
        )
        self.season = season
        self.episode = episode
        self.connection_character = connection_character
        self.context_size = context_size

    def tokenize_and_match(
        self: "PrepareTokenizedTextBatches",
    ) -> defaultdict[int, list[int]]:
        """Tokenize and map sub-tokens with the words.

        Returns:
            - mapping: A dictionary {int: list(int)} mapping of
            untokenized words with the subtokens.
        """
        eos_token = "<|endoftext|>"
        untokenized_text = preprocess_words(self.tsv_path)
        tokenized_text = self.tokenizer.tokenize(untokenized_text)
        mapping = defaultdict(list)
        untokenized_text_index = 0
        tokenized_text_index = 0
        while untokenized_text_index < len(
            untokenized_text,
        ) and tokenized_text_index < len(tokenized_text):
            while (
                tokenized_text_index + 1 < len(tokenized_text)
                and (
                    not tokenized_text[tokenized_text_index + 1].startswith(
                        self.connection_character,
                    )
                )
                and tokenized_text[tokenized_text_index + 1] != eos_token
            ):
                mapping[untokenized_text_index].append(tokenized_text_index)
                tokenized_text_index += 1
            mapping[untokenized_text_index].append(untokenized_text_index)
            untokenized_text_index += 1
            tokenized_text_index += 1
        return mapping

    def pad_to_max_length(
        self: "PrepareTokenizedTextBatches",
        sequence: list[Any],
        padding_value: Any = 220,
        end_token: Any = 50256,
    ) -> Any:
        """Pag sequence to reach max_seq_length.

        Args:
            - sequence: text chunked in context size
            - padding_value: int (default 220)
            - end_token: int (default 50256)

        Returns:
            - result: padded texts as list of int
        """
        sequence = sequence[: self.max_seq_length]
        padding_required = self.max_seq_length - len(sequence)
        result = sequence + [padding_value, end_token] * (
            padding_required // 2
        )
        if len(result) == self.max_seq_length:
            return result
        else:
            return result + [padding_value]

    def create_examples(
        self: "PrepareTokenizedTextBatches",
        sequence: list[Any],
        padding_value: Any = 220,
        start_token: Any = 50256,
        end_token: Any = 50256,
    ) -> Any:
        """Returns list of InputExample objects.

        Args:
            - sequence: text batched in context size
            - padding_value: int (default 220)
            - end_token: int (default 50256)
            - end_token: int (default 50256)

        Returns:
            - result: padded texts as list of int
            - start_token: int (default 50256)
            - end_token: int (default 50256)
        """
        new_sequence = [start_token] + sequence + [padding_value, end_token]
        return self.pad_to_max_length(new_sequence)

    def get_text_batches(
        self: "PrepareTokenizedTextBatches",
        padding_value: Any = 220,
        start_token: str = "<|endoftext|>",
        end_token: str = "<|endoftext|>",
    ) -> Any:
        """."""
        stimuli = preprocess_words(self.tsv_path)

        if self.context_size is None:
            self.max_seq_length = self.max_seq_length
        else:
            self.max_seq_length = (
                self.context_size + 5
            )  # count for special tokens
        try:
            data = self.tokenizer.encode(stimuli).ids
            text = self.tokenizer.encode(stimuli).tokens
        except:
            data = self.tokenizer.encode(stimuli)
            text = self.tokenizer.tokenize(stimuli)

        if self.context_size == 0:
            examples = [
                self.create_examples(data[i : i + 2])
                for i, _ in enumerate(data)
            ]
            tokens = [
                self.create_examples(
                    text[i : i + 2],
                    start_token=start_token,
                    end_token=end_token,
                    padding_value=self.connection_character,
                )
                for i, _ in enumerate(text)
            ]
        else:
            examples = [
                self.create_examples(data[i : i + self.context_size + 2])
                for i, _ in enumerate(data[: -self.context_size])
            ]
            tokens = [
                self.create_examples(
                    text[i : i + self.context_size + 2],
                    padding_value=padding_value,
                    start_token=start_token,
                    end_token=end_token,
                )
                for i, _ in enumerate(text[: -self.context_size])
            ]

        features = [
            torch.FloatTensor(example).unsqueeze(0).to(torch.int64)
            for example in examples
        ]
        input_ids = torch.cat(features, dim=0)
        indexes = [(1, self.context_size + 2)] + [
            (self.context_size + 1, self.context_size + 2)
            for i in range(1, len(input_ids))
        ]
        del examples
        del features
        return input_ids, indexes, tokens


class ExtractEmbedding:
    """."""

    def __init__(
        self: "ExtractEmbedding",
        indexes: list[tuple[int, int]],
        input_ids: torch.Tensor,
        mapping: defaultdict[int, list[int]],
        model: nn.Module,
        data_config: DataConfigBase(),
    ) -> None:
        """."""
        self.mapping = mapping
        self.model = model
        self.bsz = data_config.bsz
        self.num_hidden_layers = data_config.num_hidden_layers
        self.feature_count = data_config.feature_count
        self.indexes = indexes
        self.input_ids = input_ids

    def get_hidden_features(
        self: "ExtractEmbedding",
    ) -> pd:
        """."""
        features = []
        with torch.no_grad():
            hidden_states_activations_ = []
            for input_tmp in tqdm(
                self.input_ids.chunk(self.input_ids.size(0) // self.bsz),
            ):
                hidden_states_activations_tmp = []
                encoded_layers = self.model(
                    input_tmp,
                    output_hidden_states=True,
                )
                hidden_states_activations_tmp = np.stack(
                    [i.detach().numpy() for i in encoded_layers.hidden_states],
                    axis=0,
                )
                # shape: (#nb_layers, batch_size_tmp, max_seq_length, hidden_state_dimension)
                hidden_states_activations_.append(
                    hidden_states_activations_tmp,
                )

            hidden_states_activations_ = np.swapaxes(
                np.vstack(
                    [
                        np.swapaxes(item, 0, 1)
                        for item in hidden_states_activations_
                    ],
                ),
                0,
                1,
            )
        # shape: (#nb_layers, batch_size, max_seq_length, hidden_state_dimension)
        activations = []
        for i in range(hidden_states_activations_.shape[1]):
            index = self.indexes[i]
            activations.append(
                [
                    hidden_states_activations_[:, i, j, :]
                    for j in range(index[0], index[1])
                ],
            )
        activations = np.stack([i for l in activations for i in l], axis=0)
        activations = np.swapaxes(activations, 0, 1)
        # shape: (#nb_layers, batch_size, hidden_state_dimension)

        for word_index in range(len(self.mapping.keys())):
            word_activation = []
            word_activation.append(
                [
                    activations[:, index, :]
                    for index in self.mapping[word_index]
                ],
            )  # this is just a list of stacked lists
            word_activation = np.vstack(
                word_activation,
            )  # this stucks lists of actications of sub-tokens of a word and convert it into a numpy array

            features.append(
                np.mean(word_activation, axis=0).reshape(-1)
            )  # here is a list keep appending
            # mean over subtokens of a word taken to estimate the word activation.

            # list of elements of shape:
            # (#nb_layers, hidden_state_dimension).reshape(-1)
            # After vstacking it will be of shape:
            # (batch_size, #nb_layers*hidden_state_dimension)

        return pd.DataFrame(
            np.vstack(features),  # converts to a numpy array
            columns=[
                "hidden_state-layer-{}-{}".format(layer, index)
                for layer in np.arange(1 + self.num_hidden_layers)
                for index in range(1, 1 + self.feature_count)
            ],
        )


def prepare_embeddings(
    data_config: data_config,
    season: str,
    episode: str,
    finetuned: bool = False,
) -> np:
    """Definition below.

    - Gets the list for training  and test season
    - Extracts features per episodes
    - Reads the fmri data per episode (in np format)
    - Gets the gentle files per episode
    - Bundles the fmri, features, gentles for training and test.

    Args:
        - data_config: config object for the necessary arguments
        - finetuned: True if the model is finetuned
    """
    tokenizer = AutoTokenizer.from_pretrained(data_config.base_model_name)

    if finetuned:
        model_config = BaseLitModuleConfig()
        peft_config = LoraConfig(
            task_type=data_config.task_type,
            inference_mode=data_config.inference_mode,
            r=data_config.r,
            lora_alpha=data_config.lora_alpha,
            lora_dropout=data_config.lora_dropout,
            fan_in_fan_out=data_config.fan_in_fan_out,
        )
        nnmodule = AutoModelForCausalLM.from_pretrained(
            data_config.base_model_name,
        )

        ckp_path: Path = (
            Path(data_config.data_dir)
            / data_config.base_model_name
            / f"{data_config.sweep}"
            / "lightning"
            / "last.ckpt"
        )

        model = FriendsFinetuningModel.load_from_checkpoint(
            ckp_path,
            config=model_config,
            peft_config=peft_config,
            nnmodule=nnmodule,
            optimizer=partial(Adam),
            scheduler=partial(get_constant_schedule),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            data_config.base_model_name,
        )

    text = PrepareTokenizedTextBatches(
        tokenizer=tokenizer,
        tsv_path=data_config.tsv_path,
        season=season,
        episode=episode,
        context_size=data_config.context_size,
        connection_character="Ġ",
    )

    mapping = text.tokenize_and_match()
    input_ids, indexes, tokens = text.get_text_batches()

    embeddings = ExtractEmbedding(
        indexes=indexes,
        mapping=mapping,
        input_ids=input_ids,
        model=model,
        data_config=data_config,
    )
    # features that each column is a hidden layer of the model

    return embeddings.get_hidden_features()


def prepare_embedding_h5py(
    data_config,
):

    # # prepare hdf5 bundles that includes extracted features
    # # and fmri time series per episodes

    seasons = list_seasons(data_config.tsv_path)
    print(seasons)

    # create embeddigns
    for season in seasons:
        # creates a h5py file per season including its
        # all episodes embeddings.
        comp_args, outfile_name = set_output(
            season=season,
            output_dir=data_config.output_dir,
        )
        episode_list = list_episodes(
            idir=data_config.tsv_path,
            season=season,
            outfile=outfile_name,
        )

        # Create a new HDF5 file
        outfile = (
            Path(data_config.stimuli_dir) / f"friends_{season}_embeddings.h5"
        )
        with h5py.File(outfile, "w") as file:
            for episode in episode_list:
                print(episode)
                feature = prepare_embeddings(
                    data_config=data_config,
                    season=season,
                    episode=episode,
                    finetuned=False,
                )
                file.create_dataset(episode, data=feature)


def prepare_embedding_pkl(
    data_config,
):

    # # prepare hdf5 bundles that includes extracted features
    # # and fmri time series per episodes

    seasons = list_seasons(data_config.tsv_path)
    print(seasons)

    # create embeddigns
    for season in seasons:
        features = pd.DataFrame()
        # creates a h5py file per season including its
        # all episodes embeddings.
        comp_args, outfile_name = set_output(
            season=season,
            output_dir=data_config.stimuli_dir,
        )
        episode_list = list_episodes(
            idir=data_config.tsv_path,
            season=season,
            outfile=outfile_name,
        )
        outfile = (
            Path(data_config.stimuli_dir) / f"friends_{season}_embeddings.pkl"
        )

        for episode in episode_list:
            print(episode)
            features[episode] = prepare_embeddings(
                data_config=data_config,
                season=season,
                episode=episode,
                finetuned=False,
            )

        with open(outfile, "wb") as f:
            pickle.dump(features, f)


def get_layer_embeddding(data_config):

    seasons = list_seasons(data_config.tsv_path)
    print(seasons)
    for season in seasons:
        comp_args, outfile_name = set_output(
            season=season,
            output_dir=data_config.stimuli_dir,
        )
        episode_list = list_episodes(
            idir=data_config.tsv_path,
            season=season,
            outfile=outfile_name,
        )
        print(episode_list)
        # Create a new HDF5 file
        outfile = (
            Path(data_config.stimuli_dir)
            / f"friends_{season}_layer_{data_config.target_layer - 1}_embeddings.h5"
        )
        print(outfile)
        with h5py.File(outfile, "w") as file:
            for episode in episode_list:
                print(episode)
                feature = prepare_embeddings(
                    data_config=data_config,
                    season=season,
                    episode=episode,
                    finetuned=False,
                )

                layer_features = [
                    row[
                        (data_config.target_layer - 1)
                        * data_config.feature_count : data_config.target_layer
                        * data_config.feature_count
                    ]
                    for _, row in feature.iterrows()
                ]
                print("creating the file")

                file.create_dataset(episode, data=layer_features)


def create_train_val_test_stimuli(data_config, season: str):
    """."""

    season_stimuli = os.path.join(
        data_config.stimuli_dir,
        f"friends_{season}_layer_{data_config.target_layer-1}_embeddings.h5",
    )
    word_feature_matrix = []

    with h5py.File(season_stimuli, "r") as file:
        for name in file:
            print(name)
            print("hey")
            dataset = file[name][...]
            print(len(dataset))
            # Concatanate the word vectors from all episodes
            word_feature_matrix.extend(
                dataset,
            )

    return word_feature_matrix