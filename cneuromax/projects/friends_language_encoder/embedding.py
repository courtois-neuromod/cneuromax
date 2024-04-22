"""."""

import os
import string
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from cneuromax.projects.friends_language_encoder.data import ProcessText


class ExtractEmbedding:
    """."""

    def __init__(
        self: "ExtractEmbedding",
        indexes: list[tuple[int, int]],
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
# tsv_path = "./data/friends_language_encoder/annotations/"
tsv_path = "./data/friends_language_encoder/stimuli/"


text = ProcessText(
    tokenizer=tokenizer,
    tsv_path=tsv_path,
    season=1,
    episode="s01e01a",
    context_size=50,
    connection_character="Ġ",
)


mapping = text.tokenize_and_match()
input_ids, indexes, tokens = text.get_text_chunks()


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
