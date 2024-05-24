import glob

# Open an existing HDF5 file
# with h5py.File(
#     "/sub-03/func/sub-03_task-friends_space-MNI152NLin2009cAsym_atlas-MIST_desc-444_timeseries.h5",
#     "r",
# ) as file:
#     # List all groups
#     print("Keys: %s" % file.keys())
#     a_group_key = list(file.keys())[0]
#     # Get the data
#     data = list(file[a_group_key])
# print(data)
# dir = "./stimuli/"
# season = "s1"
# data = pd.DataFrame()
# with h5py.File(f"friends_{season}_embeddings.h5", "w") as file:
#     for x in sorted(glob.glob(f"{dir}/{season}/friends_s*.tsv")):
#         print(x)
#         episode = x.split("/")[-1].split(".")[0][8:15]
#         print(episode)
#         data[episode] = pd.read_csv(x, sep="\t")
#     file.create_dataset(episode, data=data)
# with h5py.File(f"friends_{season}_embeddings.h5", "r") as file:
#     for name in file:
#         print(name)
# data_to_add = {
#     "dataset1": np.random.rand(10, 10),
#     "dataset2": np.random.rand(20, 20),
#     "dataset3": np.random.rand(5, 5),
# }
# # Create a new HDF5 file
# with h5py.File("example.h5", "w") as file:
#     for identifier, data in data_to_add.items():
#         print(identifier)
#         # Create a dataset for each item
#         file.create_dataset(identifier, data=data)
# # Verify by reopening the file and checking contents
# with h5py.File("example.h5", "r") as file:
#     for name in file:
#         print(f"Dataset {name}:")
#         print(file[name][:])  # This prints the data in each dataset
from dataclasses import dataclass
from pathlib import Path

from cneuromax.projects.friends_language_encoder.embedding import (
    DataConfigBase,
    get_layer_embeddding,
    prepare_embeddings,
)
from cneuromax.projects.friends_language_encoder.utils import (
    list_episodes,
    split_episodes,
)

seasons = ["s01", "s02", "s03", "s04", "s05", "s06"]
print(seasons)
data_config = DataConfigBase()

# get layer embeddings

# get train and test fmri and stimuli sets.
# train_groups, train_runs, val_runs, test_runs, val_season = split_episodes(
#     data_config,
# )
print("starting embedding extraction")

get_layer_embeddding(data_config)
print("done embedding extraction")


# training_seasons = list(
#     filter(lambda x: x not in [data_config.test_season, val_season], seasons),
# )


# # for season in ["s02"]:
# #     stimuli_file = (
# #         Path(data_config.stimuli_dir)
# #         / f"friends_{season}_layer_{data_config.target_layer - 1}_embeddings.h5"
# #     )
# #     print(stimuli_file)
# #     with h5py.File(stimuli_file, "r") as file:
# #         print("opened")
# #         for episode in file:
# #             print(file[episode])
# # #             print(episode)


# # build_text(data_config, training_seasons, train_runs)


# y_train, length_train, train_groups = build_output(
#     data_config,
#     train_runs,
#     train_groups,
# )


# # print(f"y_train_length: {len(y_train)}")
# # print(f"length_train: {length_train}")
# # print(f"train_groups: {train_groups}")
# # print(f"train_runs: {train_runs}")

# y_val, length_val, _ = build_output(
#     data_config,
#     val_runs,
# )


# # print(f"y_val_length: {len(y_val)}")
# # print(f"length_train: {length_val}")
# # print(f"y_val: {y_val}")
# # print(f"val_runs: {val_runs}")
# # # create train, val, test text
# x_train = build_text(
#     data_config,
#     train_runs,
#     length_train,
# )

# x_val = build_text(
#     data_config,
#     val_runs,
#     length_val,
# )


# model = train_ridgeReg(
#     x_train,
#     y_train,
#     train_groups,
#     data_config,
# )

# test_ridgeReg(
#     data_config,
#     model,
#     x_train,
#     y_train,
#     x_val,
#     y_val,
# )


# test_stimuli = create_train_val_test_stimuli(data_config, test_season)
# val_stimuli = create_train_val_test_stimuli(data_config, val_season)
# training_stimuli = []
# for train in training_seasons:
#     training_stimuli = [
#         training_stimuli,
#         create_train_val_test_stimuli(data_config, train),
#     ]


# print(len(training_stimuli))
# print(len(training_stimuli[0]))


# print(test)


# for seasin


# # prepare hdf5 bundles that includes extracted features
# # and fmri time series per episodes

# test_set = []
# with h5py.File(
#     Path(data_config.fmri_dir) / data_config.fmri_file,
#     "r",
# ) as file:
#     for ses in file:
#         test_set += [x for x in file[ses] if x.split("-")[-1][:3] == "s03"]

# test_set = "s3"

# test_stimuli = (
#     Path(data_config.stimuli_dir) / f"friends_{test_set}_embeddings.h5"
# )
# layer_number = 5  # change this to whatever layer number you need

# # with h5py.File(test_stimuli, "r") as file:
#     for episode in file:
#         print(episode)
#         data = file[episode][...]
#         print("Shape of the dataset:", data.shape)

# for i in range(data.shape[1]):  # Assuming columns are features
#     column_data = data[:, i]
#     print(f"Data in column {i}: {column_data}")

# # Let's say you want to extract embeddings from layer 5 (assuming layer numbering starts from 1)
# layer_columns = [
#     col
#     for col in data.columns
#     if f"hidden_state-layer-{layer_number}-" in col
# ]
# layer_embeddings = data[layer_columns]
# print(layer_embeddings)


# seasons = list_seasons(data_config.tsv_path)
# print(seasons)
# for season in seasons:
#     comp_args, outfile_name = set_output(
#         season=season,
#         output_dir=data_config.output_dir,
#     )
#     episode_list = list_episodes(
#         idir=data_config.tsv_path,
#         season=season,
#         outfile=outfile_name,
#     )

#     for episode in episode_list:

## load fmri


## split train-test

## run ridge


# split_episodes(
#     bold_dir=data_config.fmri_dir,
#     stimuli_dir=data_config.stimuli_dir,
#     atlas="MIST",
#     parcel="444",
#     subject_id="sub-03",
#     n_split=4,
#     random_state=42,
# )


# test_file = (
#     Path(data_config.stimuli_dir) / f"friends_{test_season}_embeddings.h5"
# # )

# import os

# test_file = os.path.join(
#     data_config.stimuli_dir,
#     f"friends_{test_season}_layer_{data_config.target_layer}embeddings.h5",
# )
# print(test_file)

# import numpy as np

# with h5py.File(test_file, "r") as file:
#     for name in file:
#         print(name)
#         dataset = file[name][...]
#         # Convert the dataset to a NumPy array
#         print(dataset)
#         print(type(dataset))
#         print(len(dataset[0]))
#         print(len(dataset[10]))

#         # data = np.array(dataset)

#         # # Create a DataFrame from the NumPy array
#         # df = pd.DataFrame(data, columns=dataset.attrs["column_names"])
#         # print(df)


# layer_embedding


# prepare hdf5 bundles that includes extracted features
# and fmri time series per episodes

# seasons = list_seasons(data_config.tsv_path)
# print(seasons)
# for season in seasons[1:]:
#     comp_args, outfile_name = set_output(
#         season=season,
#         output_dir=data_config.stimuli_dir,
#     )
#     episode_list = list_episodes(
#         idir=data_config.tsv_path,
#         season=season,
#         outfile=outfile_name,
#     )
#     # Create a new HDF5 file
#     outfile = Path(data_config.stimuli_dir) / f"friends_{season}_embeddings.h5"

#     for episode in episode_list[:1]:
#         print(episode)
#         feature = prepare_embeddings(
#             data_config=data_config,
#             season=season,
#             episode=episode,
#             finetuned=False,
#         )

#     target_layer = 2  # Choose the layer number (e.g., 2 for the second layer)
#     layer_features = [
#         row[
#             (target_layer - 1)
#             * data_config.feature_count : target_layer
#             * data_config.feature_count
#         ]
#         for _, row in feature.iterrows()
#     ]

#     print(layer_features[0].values.flatten())

# data_vector = layer_features.values.flatten()
# print(data_vector)

# file.create_dataset(episode, data=feature)

# with h5py.File(f"friends_{season}_embeddings.h5", "r") as file:
#     for name in file:
#         print(name)


## load fmri


## split train-test

## run ridge


# get_layer_embeddding(data_config)
# print(layer_features[0].values.flatten())
