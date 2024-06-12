from dataclasses import dataclass
from pathlib import Path

from cneuromax.projects.friends_language_encoder.embedding import (
    DataConfigBase, get_layer_embedding)

seasons = ["s01", "s02", "s03", "s04", "s05", "s06"]
print(seasons)
data_config = DataConfigBase()

# get layer embeddings

# get train and test fmri and stimuli sets.
# train_groups, train_runs, val_runs, test_runs, val_season = split_episodes(
#     data_config,
# )
print("starting embedding extraction")

get_layer_embedding(data_config)
print("done embedding extraction")
