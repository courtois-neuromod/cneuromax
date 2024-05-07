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

from cneuromax.projects.friends_language_encoder.embedding import (
    DataConfigBase,
    create_train_val_test_stimuli,
    get_layer_embeddding,
    prepare_embedding_pkl,
    prepare_embeddings,
)
from cneuromax.projects.friends_language_encoder.utils import (
    build_input,
    build_output,
    build_text,
    list_episodes,
    split_episodes,
    test_ridgeReg,
    train_ridgeReg,
)


@dataclass
class DataConfig(DataConfigBase):
    """."""

    bold_dir: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/fmri_data/"
    )
    stimuli_dir: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/gpt2"
    )
    output_dir: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/ridge_regression"
    )
    tsv_path: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/"
    )
    fmri_file: str = (
        "sub-03_task-friends_space-MNI152NLin2009cAsym_atlas-MIST_desc-444_timeseries.h5"
    )
    target_layer: int = 13
    atlas: str = "MIST"
    parcel: str = "444"
    subject_id: str = "sub-03"
    n_split: int = 7
    random_state: int = 42
    test_season = "s03"  # season allocated for test
    TR_delay = (
        5  # "How far back in time (in TRs) does the input window start "
    )
    # "in relation to the TR it predicts. E.g., back = 5 means that input "
    # "features are sampled starting 5 TRs before the target BOLD TR onset",
    duration = 3
    # "Duration of input time window (in TRs) to predict a BOLD TR. "
    # "E.g., input_duration = 3 means that input is sampled over 3 TRs "
    # "to predict a target BOLD TR.",


seasons = ["s01", "s02", "s03", "s04", "s05", "s06"]

data_config = DataConfig()

# get layer embeddings

# get train and test fmri and stimuli sets.
train_groups, train_runs, val_runs, test_runs, val_season = split_episodes(
    data_config,
)

# get_layer_embeddding(data_config)


training_seasons = list(
    filter(lambda x: x not in [data_config.test_season, val_season], seasons),
)


# for season in ["s02"]:
#     stimuli_file = (
#         Path(data_config.stimuli_dir)
#         / f"friends_{season}_layer_{data_config.target_layer - 1}_embeddings.h5"
#     )
#     print(stimuli_file)
#     with h5py.File(stimuli_file, "r") as file:
#         print("opened")
#         for episode in file:
#             print(file[episode])
# #             print(episode)


# build_text(data_config, training_seasons, train_runs)


y_train, length_train, train_groups = build_output(
    data_config,
    train_runs,
    train_groups,
)


# print(f"y_train_length: {len(y_train)}")
# print(f"length_train: {length_train}")
# print(f"train_groups: {train_groups}")
# print(f"train_runs: {train_runs}")

y_val, length_val, _ = build_output(
    data_config,
    val_runs,
)


# print(f"y_val_length: {len(y_val)}")
# print(f"length_train: {length_val}")
# print(f"y_val: {y_val}")
# print(f"val_runs: {val_runs}")
# # create train, val, test text
x_train = build_text(
    data_config,
    train_runs,
    length_train,
)

x_val = build_text(
    data_config,
    val_runs,
    length_val,
)


model = train_ridgeReg(
    x_train,
    y_train,
    train_groups,
    data_config,
)

test_ridgeReg(
    data_config,
    model,
    x_train,
    y_train,
    x_val,
    y_val,
)
