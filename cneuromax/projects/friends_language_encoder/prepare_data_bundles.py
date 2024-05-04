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
)


@dataclass
class DataConfig(DataConfigBase):
    """."""

    fmri_dir: str = "./fmri_data/friends_language_encoder/"
    target_layer: int = 12


import os

data_config = DataConfig()



    # Build X arrays from input features
    x_train = build_input(
        args.idir,
        args.modalities,
        args.text_features,
        train_runs,
        length_train,
        args.input_duration,
    )
    x_val = build_input(
        args.idir,
        args.modalities,
        args.text_features,
        val_runs,
        length_val,
        args.input_duration,
    )

    # Train ridge regression model on train set
    model = train_ridgeReg(
        x_train,
        y_train,
        train_groups,
        args.n_split,
    )

    # Test model and export performance metrics
    test_ridgeReg(
        args.odir,
        args.atlas,
        args.parcel,
        args.participant,
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        args.modalities,
        args.text_features,
        args.bdir,
    )





















# split_episodes(
#     bold_dir=data_config.fmri_dir,
#     stimuli_dir=data_config.stimuli_dir,
#     atlas="MIST",
#     parcel="444",
#     subject_id="sub-03",
#     n_split=4,
#     random_state=42,
# )

# test_season = "s3"

# test_file = os.path.join(
#     data_config.stimuli_dir,
#     f"friends_{test_season}_embeddings.h5",
# )

# # layer_embedding

# with h5py.File(test_file, "r") as file:
#     # List all datasets in the file
#     for episode in file:
#         # Load the dataset for the selected episode
#         episode_data = file[episode][...]

#         # Now episode_data is a numpy array with each column corresponding to a feature
#         # Let's print the shape of this dataset
#         print("Shape of the dataset:", episode_data.shape)

#         # Iterate over columns (features)
#         for i in range(episode_data.shape[1]):  # Assuming columns are features
#             column_data = episode_data[:, i]
#             print(f"Data in column {i}: {column_data}")


# print(feature)
# layer_features = [
#     row[-data_config.feature_count :] for _, row in feature.iterrows()
# ]


# layer_features = [
#     row[
#         (data_config.target_layer - 1)
#         * data_config.feature_count : data_config.target_layer
#         * data_config.feature_count
#     ]
#     for _, row in feature.iterrows()
# ]
# word_feature = pd.DataFrame()
# for word in word_list:
#     print(word)
#     word_feature[word] = layer_features[word].values.flatten()
# print(word_feature)
# with h5py.File(test_file, "w") as file:


## load fmri


## split train-test

## run ridge
