import string
from dataclasses import dataclass
from pathlib import Path

import h5py
import pandas as pd

from cneuromax.projects.friends_language_encoder.embedding import (
    DataConfigBase,
    PrepareTokenizedTextBatches,
)
from cneuromax.projects.friends_language_encoder.utils import (
    preprocess_words,
    split_episodes,
)


@dataclass
class DataConfig(DataConfigBase):
    """."""

    fmri_dir: str = "./fmri_data/friends_language_encoder/"
    target_layer: int = 12


data_config = DataConfig()


split_episodes(
    bold_dir=data_config.fmri_dir,
    stimuli_dir=data_config.stimuli_dir,
    atlas="MIST",
    parcel="444",
    subject_id="sub-03",
    n_split=4,
    random_state=42,
)

test_season = "s3"

test_file = (
    Path(data_config.stimuli_dir) / f"friends_{test_season}_embeddings.h5"
)


with h5py.File(test_file, "r") as file:
    for episode in file:
        tsv_path = Path(data_config.tsv_path) / f"friends_{episode}.tsv"
        word_list = preprocess_words(tsv_path)
        feature = file[episode]
        # get each word's embedding over all layers
        layer_features = [
            row[-data_config.feature_count :] for _, row in feature.iterrows()
        ]
        layer_features = [
            row[
                (data_config.target_layer - 1)
                * data_config.feature_count : data_config.target_layer
                * data_config.feature_count
            ]
            for _, row in feature.iterrows()
        ]
        word_feature = pd.DataFrame()
        for word in word_list:
            word_feature[word] = layer_features[word].values.flatten()


## load fmri


## split train-test

## run ridge
