"""``project`` utilities."""

import glob
import json
import logging
import os
import string
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml
from datasets.formatting.formatting import LazyBatch
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, pipeline


def group_texts_hf(examples: Any, block_size: Any) -> Any:  # noqa:ANN401,D103
    # Concatenate all texts.
    concatenated_examples: Any = {
        k: sum(examples[k], []) for k in examples.keys()  # noqa: SIM118
    }
    total_length = len(
        concatenated_examples[list(examples.keys())[0]],  # noqa: RUF015
    )
    # We drop the small remainder, we could add padding if the model
    # supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def group_texts(
    text: LazyBatch,
    chunk_size: int,
) -> dict[str, list[str]]:
    """.

    Concatanates and chunks the data into custom chunk size
    (mostly max_seq_length)

    Args:
      text: text data in huggingface dataset format
      chunk_size: custom size of the word sequences

    """
    # Concatenate all texts
    concatenated_text: Any = {key: sum(text[key], []) for key in text}
    # Compute length of concatenated texts
    total_length = len(
        concatenated_text[list(text.keys())[0]],  # noqa: RUF015
    )
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [
            t[i : i + chunk_size]
            for i in range(
                0,
                total_length - chunk_size + 1,
                1,
            )  # 1 is overlap
        ]
        for k, t in concatenated_text.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


logging.basicConfig(filename="loggings.log", level=logging.INFO)


def check_folder(path: str) -> None:
    """Create adequate folders if necessary.

    Args:
        - path: str
    """
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass


def read_tsv(tsv_path: str) -> pd.DataFrame:
    """."""
    return pd.read_csv(tsv_path, sep="\t")


def read_yaml(yaml_path: str) -> Any:
    """Open and read safely a yaml file.

    Args:
        - yaml_path: str
    Returns:
        - parameters: dict
    """
    try:
        with open(yaml_path, "r") as stream:
            parameters = yaml.safe_load(stream)
        return parameters
    except:
        print("Couldn't load yaml file: {}.".format(yaml_path))


def save_yaml(data: Any, yaml_path: str) -> None:
    """Open and write safely in a yaml file.

    Args:
        - data: list/dict/str/int/float
        - yaml_path: str
    """
    with open(yaml_path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def write(path: str, text: str, end: str = "\n") -> None:
    """Write in the specified text file.

    Args:
        - path: str
        - text: str
        - end: str
    """
    with open(path, "a+") as f:
        f.write(text)
        f.write(end)


class Identity(PCA):
    def __init__(self: "Identity") -> None:
        """Implement identity operator."""
        pass

    def fit(self: "Identity", X: Any, y: Any) -> None:
        pass

    def transform(self: "Identity", X: Any) -> Any:
        return X

    def fit_transform(self: "Identity", X: Any, y: Any = None) -> Any:
        self.fit(X, y=y)
        return self.transform(X)

    def inverse_transform(self: "Identity", X: Any) -> Any:
        return X


def get_possible_linear_models() -> list[str]:
    """Fetch possible reduction methods.

    Returns:
        - list
    """
    return ["ridgecv", "glm"]


def get_possible_reduction_methods() -> list[str]:
    """Fetch possible reduction methods.

    Returns:
        - list
    """
    return [None, "pca", "agglomerative_clustering"]


STUDY_PARAMS = {
    "tr": 1.49,
    "max_tokens": 512,
}


def list_seasons(
    idir: str,
) -> list:
    """."""

    season_list = [
        x.split("/")[-1] for x in sorted(glob.glob(f"{idir}/s[0-9]"))
    ]

    return season_list


def list_episodes(
    idir: str,
    season: str,
    outfile: str,
) -> list:
    """.

    Compile season's list of episodes to process.
    """
    all_epi = [
        x.split("/")[-1].split(".")[0][8:15]
        for x in sorted(glob.glob(f"{idir}/{season}/friends_s*.tsv"))
    ]

    if Path(outfile).exists():
        season_h5_file = h5py.File(outfile, "r")
        processed_epi = list(season_h5_file.keys())
        season_h5_file.close()
    else:
        processed_epi = []

    episode_list = [epi for epi in all_epi if epi not in processed_epi]

    return episode_list


def set_output(
    season: str,
    output_dir: str,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> tuple:
    """.

    Set compression params and output file name.
    """
    compress_details = ""
    comp_args = {}
    if compression is not None:
        compress_details = f"_{compression}"
        comp_args["compression"] = compression
        if compression == "gzip":
            compress_details += f"_level-{compression_opts}"
            comp_args["compression_opts"] = compression_opts

    out_file = (
        f"{output_dir}/friends_{season}_features_" f"text{compress_details}.h5"
    )

    # Path(f"{args.odir}/temp").mkdir(exist_ok=True, parents=True)

    return comp_args, out_file


def save_features(
    episode: str,
    feature: np.array,
    outfile_name: str,
    comp_args: dict,
) -> None:
    """.

    Save episode's text features into .h5 file.
    """
    flag = "a" if Path(outfile_name).exists() else "w"

    with h5py.File(outfile_name, flag) as f:
        group = f.create_group(episode)

        group.create_dataset(
            "features",
            data=feature,
            **comp_args,
        )


def preprocess_words(tsv_path:str) -> str:
    """Un-punctuate, lower and combine the words like a text.

    Args:
        - tsv_path: path to the episode file
    Returns:
        - list of concatanated words
    """
    data = read_tsv(tsv_path)
    stimuli_data = data["word"].apply(
        lambda x: x.translate(
            str.maketrans("", "", string.punctuation),
        ).lower(),
    )

    return " ".join(stimuli_data)


# def split_episodes(
#     bold_dir: str,
#     stimuli_dir: str,
#     atlas: str,
#     parcel: str,
#     subject_id: str,
#     n_split: int,
#     random_state: int = None,
#     test_set: str = "s03",
#     seasons: list[str] = ["s01", "s02", "s03", "s04", "s05", "s06"],
#     seasons_sub_04: list[str] = ["s01", "s02", "s03", "s04"],
# ) -> tuple:
#     """.

#     Assigns subject's runs to train, validation and test sets
#     """
#     sub_h5_fmri = h5py.File(
#         f"{bold_dir}/{atlas}_{parcel}/{subject_id}/func/"
#         f"{subject_id}_task-friends_space-MNI152NLin2009cAsym_"
#         f"atlas-{atlas}_desc-{parcel}_timeseries.h5",
#         "r",
#     )

#     # Season 3 held out for test set
#     test_fmri_set = []
#     for ses in sub_h5_fmri:
#         test_fmri_set += [
#             x for x in sub_h5_fmri[ses] if x.split("-")[-1][:3] == test_set
#         ]


#     test_stimuli= Path(data_config.stimuli_dir) / f"friends_{test_set}_embeddings.h5"

#     with h5py.File(test_stimuli, "r") as file:
#         for episodes in file:


# # Remaining runs assigned to train and validation sets
# r = np.random.RandomState(random_state)  # select season for validation set

# if subject_id == "sub-04":
#     val_season = r.choice(seasons_sub_04.remove(test_set), 1)[0]
# else:
#     val_season = r.choice(seasons.remove(test_set), 1)[0]
# val_fmri_set = []
# for ses in sub_h5_fmri:
#     val_fmri_set += [
#         x for x in sub_h5_fmri[ses] if x.split("-")[-1][:3] == val_season
#     ]
# train_fmri_set = []
# for ses in sub_h5_fmri:
#     train_fmri_set += [
#         x
#         for x in sub_h5_fmri[ses]
#         if x.split("-")[-1][:3] not in ["s03", val_season]
#     ]
# train_fmri_set = sorted(train_fmri_set)

# sub_h5_fmri.close()

# # Assign consecutive train set episodes to cross-validation groups
# lts = len(train_fmri_set)
# train_groups = (
#     np.floor(np.arange(lts) / (lts / n_split)).astype(int).tolist()
# )


# return train_groups, train_fmri_set, val_fmri_set, test_fmri_set


# def get_linearmodel(name, alpha=1, alpha_min=-3, alpha_max=8, nb_alphas=10):
#     """Retrieve the"""
#     if name == "ridgecv":
#         logging.info(
#             f"Loading RidgeCV, with {nb_alphas} alphas varying logarithimicly between {alpha_min} and {alpha_max}..."
#         )
#         return RidgeCV(
#             np.logspace(alpha_min, alpha_max, nb_alphas),
#             fit_intercept=True,
#             alpha_per_target=True,
#             scoring="r2",
#         )
#     elif name == "glm":
#         logging.info(f"Loading LinearRegression...")
#         return LinearRegression(fit_intercept=True)
#     elif not isinstance(name, str):
#         logging.warning(
#             "The model seems to be custom.\nUsing it directly for the encoding analysis."
#         )
#         return name
#     else:
#         logging.error(
#             f"Unrecognized model {name}. Please select among ['ridgecv', 'glm] or a custom encoding model."
#         )


# def get_reduction_method(method, ndim=None):
#     """
#     Args:
#         - method: str
#         - ndim: int
#     Returns:
#         - output: built-in reduction operator
#     """
#     if method is None:
#         return Identity()
#     elif method == "pca":
#         return PCA(n_components=ndim)
#     elif method == "agglomerative_clustering":
#         return FeatureAgglomeration(n_clusters=ndim)


# def get_groups(gentles):
#     """Compute the number of rows in each array
#     Args:
#         - gentles: list of np.Array
#     Returns:
#         - groups: list of np.Array
#     """
#     # We compute the number of rows in each array.
#     lengths = [len(f) for f in gentles]
#     start_stop = []
#     start = 0
#     for l in lengths:
#         stop = start + l
#         start_stop.append((start, stop))
#         start = stop
#     groups = [np.arange(start, stop, 1) for (start, stop) in start_stop]
#     return groups
