"""``project`` utilities."""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml
from datasets.formatting.formatting import LazyBatch
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV


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
