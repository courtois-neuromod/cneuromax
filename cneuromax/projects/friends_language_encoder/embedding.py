"""."""

import os
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    pipeline,
)


class PrepareStimuli:
    """."""

    def __init__(
        self: "PrepareStimuli",
        tokenizer: PreTrainedTokenizerBase,
        tsv_path: str,
        season: int,
        episode: str,
        context_size: int,
        connection_character: str = "Ġ",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = tokenizer.model_max_length
        self.tsv_path = tsv_path
        self.season = season
        self.episode = episode
        self.connection_character = connection_character
        self.context_size = context_size

    def read_tsv(self: "PrepareStimuli") -> pd:
        """."""
        file = os.path.join(
            self.tsv_path,
            f"s{self.season}",
            f"friends_{self.episode}.tsv",
        )
        return pd.read_csv(file, sep="\t")

    def combine_words(self: "PrepareStimuli") -> str:
        """."""
        stimuli_data = self.read_tsv()
        stimuli_data["clean_word"] = stimuli_data["word"].apply(
            lambda x: x.translate(
                str.maketrans("", "", string.punctuation),
            ).lower(),
        )
        return " ".join(stimuli_data["clean_word"])

    def tokenize_and_match(
        self: "PrepareStimuli",
    ) -> defaultdict[int, list[int]]:
        """."""
        eos_token = "<|endoftext|>"
        untokenized_sent = self.combine_words()
        tokenized_sent = self.tokenizer.tokenize(untokenized_sent)
        mapping = defaultdict(list)
        untokenized_sent_index = 0
        tokenized_sent_index = 0
        while untokenized_sent_index < len(
            untokenized_sent,
        ) and tokenized_sent_index < len(tokenized_sent):
            while (
                tokenized_sent_index + 1 < len(tokenized_sent)
                and (
                    not tokenized_sent[tokenized_sent_index + 1].startswith(
                        self.connection_character,
                    )
                )
                and tokenized_sent[tokenized_sent_index + 1] != eos_token
            ):
                mapping[untokenized_sent_index].append(tokenized_sent_index)
                tokenized_sent_index += 1
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            untokenized_sent_index += 1
            tokenized_sent_index += 1
        return mapping

    def pad_to_max_length(
        self: "PrepareStimuli",
        sequence: list[str],
        space: Any = 220,
        special_token_end: Any = 50256,
    ) -> Any:
        """."""

        sequence = sequence[:max_seq_length]
        n = len(sequence)
        result = sequence + [space, special_token_end] * (
            (self.max_seq_length - n) // 2
        )
        if len(result) == self.max_seq_length:
            return result
        else:
            return result + [space]

    def create_examples(
        self: "PrepareStimuli",
        sequence: list[Any],
        space: Any = 220,
        special_token_beg: Any = 50256,
        special_token_end: Any = 50256,
    ) -> Any:
        """."""
        new_sequence = (
            [special_token_beg] + sequence + [space, special_token_end]
        )
        return self.pad_to_max_length(new_sequence)

    def get_text_chunks(
        self: "PrepareStimuli",
        special_token_beg: str = "<|endoftext|>",
        special_token_end: str = "<|endoftext|>",
    ) -> Any:
        """."""
        stimuli = self.combine_words()

        if self.context_size is None:
            self.max_seq_length = self.max_seq_length
        else:
            self.max_seq_length = (
                self.context_size + 5
            )  # count for special tokens
        try:
            data = tokenizer.encode(stimuli).ids
            text = tokenizer.encode(stimuli).tokens
        except:
            data = tokenizer.encode(stimuli)
            text = tokenizer.tokenize(stimuli)

        if self.context_size == 0:
            examples = [
                self.create_examples(data[i : i + 2])
                for i, _ in enumerate(data)
            ]
            tokens = [
                self.create_examples(
                    text[i : i + 2],
                    space=self.connection_character,
                    special_token_beg=special_token_beg,
                    special_token_end=special_token_end,
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
                    space=space,
                    special_token_beg=special_token_beg,
                    special_token_end=special_token_end,
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
        indexes: list[list[int]],
        input_ids: torch.Tensor,
        mapping: defaultdict[int, list[int]],
        model: nn.Module,
        bsz: int = 32,
        feature_count: int = 768,
        num_hidden_layers: int = 12,
    ) -> None:
        """."""
        self.mapping = mapping
        self.model = model
        self.bsz = bsz
        self.num_hidden_layers = num_hidden_layers
        self.feature_count = feature_count
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
                    hidden_states_activations_tmp
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
            )
            word_activation = np.vstack(word_activation)
            features.append(np.mean(word_activation, axis=0).reshape(-1))
            # list of elements of shape: (#nb_layers, hidden_state_dimension).reshape(-1)
            # After vstacking it will be of shape: (batch_size, #nb_layers*hidden_state_dimension)

        return pd.DataFrame(
            np.vstack(features),
            columns=[
                "hidden_state-layer-{}-{}".format(layer, index)
                for layer in np.arange(1 + self.num_hidden_layers)
                for index in range(1, 1 + self.feature_count)
            ],
        )


max_seq_length = 512  # maximum input size that can be given to the model
space = "Ġ"  # specific to the tokenizer...
special_token_beg = "<|endoftext|>"  # special tokens added at the beginning of the sentence specific to the tokenizer...
special_token_end = "<|endoftext|>"  # special tokens added at the end of the sentence specific to the tokenizer...


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tsv_path = "./data/friends_language_encoder/annotations/"
my_stimuli = PrepareStimuli(
    tokenizer=tokenizer,
    tsv_path=tsv_path,
    season=1,
    episode="s01e01a",
    context_size=50,
    connection_character="Ġ",
)


mapping = my_stimuli.tokenize_and_match()
input_ids, indexes, tokens = my_stimuli.get_text_chunks()


embeddings = ExtractEmbedding(
    indexes=indexes,
    mapping=mapping,
    input_ids=input_ids,
    model=model,
    bsz=32,
    feature_count=768,
    num_hidden_layers=12,
)

features = embeddings.get_hidden_features()
print(features)


# class PrepareStimuli:
#     """."""

#     def __init__(
#         self: "PrepareStimuli",
#         nnmodule: nn.Module,
#         tokenizer: PreTrainedTokenizerBase,
#         tsv_path: str,
#         season: int,
#         episode: str,
#     ) -> None:
#         self.nnmodule = nnmodule
#         self.tokenizer = tokenizer
#         self.tsv_path = tsv_path
#         self.embeddings: dict[str, Tensor] = {}
#         self.position: dict[str, Tensor] = {}
#         self.season = season
#         self.episode = episode

#     def read_tsv(self) -> pd:
#         """."""
#         file = os.path.join(self.tsv_path, f"s{self.season}", self.episode)
#         stimuli_data = pd.read_csv(self.tsv_path, sep="\t")
#         return stimuli_data

#     def combine_words(self) -> list[str]:
#         """."""
#         stimuli_data = self.read_tsv()
#         return " ".join(stimuli_data["word"].values)

#     def get_embeddings(
#         self: "PrepareStimuli",
#     ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
#         """."""
#         for i in range(df.shape[0]):
#             num_tokens = 0
#             if not df.iloc[i]["is_na"]:
#                 tr_text = df.iloc[i]["words_per_tr"]

#             text_index = self.tokenizer.encode(item, add_prefix_space=True)
#             self.embeddings[item] = self.nnmodule.transformer.wte.weight[
#                 text_index,
#                 :,
#             ]
#             self.position[item] = self.nnmodule.transformer.wpe.weight[
#                 text_index,
#                 :,
#             ]

#         return (self.embeddings, self.position)


# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# my_feature = ExtractFeatures(
#     tsv_path="s1/friends_s01e01a_results_aa.tsv",
#     nnmodule=model,
#     tokenizer=tokenizer,
# )
