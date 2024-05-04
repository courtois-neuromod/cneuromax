"""``project`` utilities."""

import argparse
import glob
import json
import logging
import os
import string
import sys
from pathlib import Path
from typing import Any

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml
from datasets.formatting.formatting import LazyBatch
from nilearn.maskers import NiftiLabelsMasker
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
        x.split("/")[-1] for x in sorted(glob.glob(f"{idir}/s0[0-9]"))
    ]

    return season_list


def list_episodes(
    idir: str,
    season: str,
    outfile: str = None,
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


def preprocess_words(tsv_path: str) -> str:
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


def split_episodes(
    data_config,
) -> tuple:
    """.

    Assigns subject's runs to train, validation and test sets
    """
    sub_h5 = h5py.File(
        f"{data_config.bold_dir}/{data_config.subject_id}/func/"
        f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
        f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
        "r",
    )

    # Season 3 held out for test set
    test_set = []
    for ses in sub_h5:
        test_set += [
            x
            for x in sub_h5[ses]
            if x.split("-")[-1][:3] == data_config.test_season
        ]

    # Remaining runs assigned to train and validation sets
    r = np.random.RandomState(
        data_config.random_state,
    )  # select season for validation set

    if data_config.subject_id == "sub-04":
        val_season = r.choice(["s01", "s02", "s04"], 1)[0]
    else:
        val_season = r.choice(["s01", "s02", "s04", "s05", "s06"], 1)[0]
    val_set = []
    for ses in sub_h5:
        val_set += [
            x for x in sub_h5[ses] if x.split("-")[-1][:3] == val_season
        ]
    train_set = []
    for ses in sub_h5:
        train_set += [
            x
            for x in sub_h5[ses]
            if x.split("-")[-1][:3]
            not in [data_config.test_season, val_season]
        ]
    train_set = sorted(train_set)

    sub_h5.close()

    # Assign consecutive train set episodes to cross-validation groups
    lts = len(train_set)
    train_groups = (
        np.floor(np.arange(lts) / (lts / data_config.n_split))
        .astype(int)
        .tolist()
    )

    return train_groups, train_set, val_set, test_set, val_season


def build_output(
    data_config,
    runs: list,
    run_groups: list = None,
) -> tuple:
    """.

    Concatenates BOLD timeseries into target array.
    """
    y_list = []
    length_list = []
    y_groups = []
    sub_h5 = h5py.File(
        f"{data_config.bold_dir}/{data_config.subject_id}/func/"
        f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_"
        f"atlas-{data_config.atlas}_desc-{data_config.parcel}_timeseries.h5",
        "r",
    )

    for i, run in enumerate(runs):
        ses = run.split("_")[0]
        run_ts = np.array(sub_h5[ses][run])[data_config.TR_delay :, :]
        length_list.append(run_ts.shape[0])
        y_list.append(run_ts)

        if run_groups is not None:
            y_groups.append(np.repeat(run_groups[i], run_ts.shape[0]))

    sub_h5.close()
    y_list = np.concatenate(y_list, axis=0)
    y_groups = (
        np.concatenate(y_groups, axis=0)
        if run_groups is not None
        else np.array([])
    )

    return y_list, length_list, y_groups


def build_text(
    data_config,
    runs: list,
    run_lengths: list,
) -> np.array:

    dur = data_config.duration - 1
    x_dict = []

    for run, rl in zip(runs, run_lengths):
        run_name = run.split("-")[-1].split("_")[0]
        season: str = run_name[:3]

        h5_path = (
            Path(data_config.stimuli_dir)
            / f"friends_{season}_layer_{data_config.target_layer - 1}_embeddings.h5"
        )

        with h5py.File(h5_path, "r") as file:
            # print(f"data = {file[run_name]}")
            # print(f"len_data = {len(file[run_name])}")
            run_data = np.array(file[run_name])[dur : dur + rl]
            # print(f"run_data = {run_data}")
            # print(f"run_data = {len(run_data)}")

            # pad features array in case fewer text TRs than for BOLD data
            rdims = run_data.shape
            # print(f"rdims = {rdims}")

            if len(rdims) == 1:
                run_data = np.expand_dims(run_data, axis=1)
                rdims = run_data.shape
            # print(f"run_data_expand = {run_data}")
            # print(f"rdims_expand = {rdims}")

            rsize = (
                rl * rdims[1] if len(rdims) == 2 else rl * rdims[1] * rdims[2]
            )
            run_array = np.repeat(np.nan, rsize).reshape((rl,) + rdims[1:])
            run_array[: rdims[0]] = run_data
            # print(f"run_array = {run_array}")

            x_dict.append(run_array)
            # print(f"x_dict = {x_dict}")

    x_list = []
    feat_data = np.concatenate(x_dict, axis=0)
    dims = feat_data.shape
    # print(f"feat_data = {feat_data}")

    x_list.append(
        np.nan_to_num(
            stats.zscore(
                feat_data.reshape((-1, dims[-1])),
                nan_policy="omit",
                axis=0,
            ),
        )
        .reshape(dims)
        .reshape(dims[0], -1)
        .astype("float32"),
    )
    # print(f"x_list = {x_list}")

    return np.concatenate(x_list, axis=1)


def build_input(
    idir: str,
    modalities: list,
    text_features: list,
    runs: list,
    run_lengths: list,
    duration: int,
) -> np.array:
    """.

    Concatenates input features across modalities into predictor array.
    """

    x_list = []

    x_list.append(
        build_text(
            idir,
            text_features,
            runs,
            run_lengths,
            duration,
        ),
    )

    if len(x_list) > 1:
        return np.concatenate(x_list, axis=1)
    else:
        return x_list[0]


def train_ridgeReg(
    X: np.array,
    y: np.array,
    groups: list,
    data_config,
) -> RidgeCV:
    """.

    Performs ridge regression with built-in cross-validation.
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
    """
    alphas = np.logspace(0.1, 3, 10)
    group_kfold = GroupKFold(n_splits=data_config.n_splits)
    cv = group_kfold.split(X, y, groups)

    model = RidgeCV(
        alphas=alphas,
        fit_intercept=True,
        # normalize=False,
        cv=cv,
    )

    return model.fit(X, y)


def pairwise_acc(
    target: np.array,
    predicted: np.array,
    use_distance: bool = False,
) -> float:
    """.

    Computes Pairwise accuracy
    Adapted from: https://github.com/jashna14/DL4Brain/blob/master/src/evaluate.py
    """
    true_count = 0
    total = 0

    for i in range(0, len(target)):
        for j in range(i + 1, len(target)):
            total += 1

            t1 = target[i]
            t2 = target[j]
            p1 = predicted[i]
            p2 = predicted[j]

            if use_distance:
                if cosine(t1, p1) + cosine(t2, p2) < cosine(t1, p2) + cosine(
                    t2, p1
                ):
                    true_count += 1

            else:
                if (
                    pearsonr(t1, p1)[0] + pearsonr(t2, p2)[0]
                    > pearsonr(t1, p2)[0] + pearsonr(t2, p1)[0]
                ):
                    true_count += 1

    return true / total


def pearson_corr(
    target: np.array,
    predicted: np.array,
) -> np.array:
    """.

    Calculates pearson R between predictions and targets.
    """
    r_vals = []
    for i in range(len(target)):
        r_val, _ = pearsonr(target[i], predicted[i])
        r_vals.append(r_val)

    return np.array(r_vals)


def export_images(
    data_config,
    results: dict,
) -> None:
    """.

    Exports RR parcelwise scores as nifti files with
    subject-specific atlas used to extract timeseries.
    """
    atlas_path = Path(
        f"{data_config.bold_dir}/{data_config.atlas}_{data_config.parcel}/{data_config.subject_id}/func/"
        f"{data_config.subject_id}_task-friends_space-MNI152NLin2009cAsym_atlas-{data_config.atlas}_"
        f"desc-{data_config.parcel}_dseg.nii.gz",
    )
    atlas_masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize=False,
    )
    atlas_masker.fit()

    # map Pearson correlations onto brain parcels
    for s in ["train", "val"]:
        nii_file = atlas_masker.inverse_transform(
            np.array(results["parcelwise"][f"{s}_R2"]),
        )
        nib.save(
            nii_file,
            f"{data_config.output_dir}/{data_config.subject_id}_{data_config.atlas}_{data_config.parcel}_RidgeReg_R2_{s}.nii.gz",
        )

    return


def test_ridgeReg(
    data_config,
    R,
    x_train,
    y_train,
    x_val,
    y_val,
) -> None:
    """.

    Exports RR results in .json file.
    """
    res_dict = {}

    # Global R2 score
    res_dict["train_R2"] = R.score(x_train, y_train)
    res_dict["val_R2"] = R.score(x_val, y_val)

    # Parcel-wise predictions
    pred_train = R.predict(x_train)
    pred_val = R.predict(x_val)

    res_dict["parcelwise"] = {}
    res_dict["parcelwise"]["train_R2"] = (
        pearson_corr(y_train.T, pred_train.T) ** 2
    ).tolist()
    res_dict["parcelwise"]["val_R2"] = (
        pearson_corr(y_val.T, pred_val.T) ** 2
    ).tolist()

    # export RR results
    Path(f"{data_config.output_dir}").mkdir(parents=True, exist_ok=True)
    with open(
        f"{data_config.output_dir}/{data_config.subject_id}_ridgeReg_{data_config.atlas}_{data_config.parcel}_result.json",
        "w",
    ) as fp:
        json.dump(res_dict, fp)

    # export parcelwise scores as .nii images for visualization
    if data_config.bold_dir is not None:
        export_images(
            data_config,
            res_dict,
        )
