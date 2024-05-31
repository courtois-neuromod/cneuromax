"""``project`` utilities."""

import glob
import logging
import os
import string
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from datasets.formatting.formatting import LazyBatch


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

def preprocess_stimuli(tsv_path: str) -> str:
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

    return stimuli_data

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

