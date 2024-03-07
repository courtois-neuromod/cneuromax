import h5py
import json
import argparse
import numpy as np
from math import ceil
from src.utils import check_list_arg


# Level categories
CATEGORIES = {
    "basic": ["w1l1", "w2l1", "w5l2", "w5l1", "w7l1", "w8l1", "w8l3"],
    "black_basic": ["w3l1", "w3l2", "w6l2"],
    "cloud": ["w4l1", "w6l1", "w8l2"],
    "fish": ["w2l3", "w7l3"],
    "underground": ["w1l2", "w4l2"],
    "platform": ["w1l3", "w5l3"],
    "pulley": ["w3l3", "w6l3", "w4l3"],
    "almost_all": [  # all but w5l1 and w6l3
        "w1l1",
        "w2l1",
        "w5l2",
        "w7l1",
        "w8l1",
        "w8l3",
        "w3l1",
        "w3l2",
        "w6l2",
        "w4l1",
        "w6l1",
        "w8l2",
        "w2l3",
        "w7l3",
        "w1l2",
        "w4l2",
        "w1l3",
        "w5l3",
        "w3l3",
        "w4l3",
    ],
    "remaining": ["w5l1", "w6l3"],
}


def split_reps(reps, tng_ratio=0.8):
    n_tng = (
        ceil(tng_ratio * len(reps))
        if len(reps) > 9
        else len(reps) - int(len(reps) > 1) - int(len(reps) > 2)
    )
    n_test = max(ceil((len(reps) - n_tng) / 2), 1) if len(reps) > 1 else 0
    tng_reps = list(reps[:n_tng])
    test_reps = list(reps[n_tng : n_tng + n_test])
    val_reps = list(reps[n_tng + n_test :])
    return tng_reps, val_reps, test_reps


def split_stratified_completed(
    mario_data_path, categories, tng_ratio=0.8, verbose=False, seed=2023
):
    """Split the mario data in train/val/test sets of runs,
    with tng_ratio being the proportion of runs kept for training, the rest goes half
    for validation and half for test, e.g. if tng_ratio is 0.8 the proportions are
    0.8/0.1/0.1, stratifying on the proportions of completed runs
    """
    categories = check_list_arg(categories, "categories", CATEGORIES.keys())
    selected_levels = []
    for cat in categories:
        selected_levels += CATEGORIES[cat]
    rng = np.random.default_rng(seed)
    mario_data = h5py.File(mario_data_path, "r")
    completed_reps = {}
    uncompleted_reps = {}

    def count_completion(name, obj):
        if "info" in name:
            sub = "sub-" + name.split("sub-")[1][:2]
            level = name.split("level-")[1][:4]
            if level in selected_levels:
                completed = -1 not in obj[:, 8]
                if sub not in completed_reps:
                    completed_reps[sub] = {}
                    uncompleted_reps[sub] = {}
                if level not in completed_reps[sub]:
                    completed_reps[sub][level] = []
                    uncompleted_reps[sub][level] = []
                if completed:
                    completed_reps[sub][level].append(name.replace("/info", ""))
                else:
                    uncompleted_reps[sub][level].append(name.replace("/info", ""))

    if verbose:
        print("counting runs ...")
    mario_data.visititems(count_completion)

    tng_list = []
    val_list = []
    test_list = []
    if verbose:
        print("subject level completed: tng/val/test uncompleted: tng/val/test")
    for sub in completed_reps:
        for level in completed_reps[sub]:
            compl_reps = rng.permutation(completed_reps[sub][level])
            compl_tng_reps, compl_val_reps, compl_test_reps = split_reps(
                compl_reps, tng_ratio
            )
            tng_list += compl_tng_reps
            val_list += compl_val_reps
            test_list += compl_test_reps
            unc_reps = rng.permutation(uncompleted_reps[sub][level])
            unc_tng_reps, unc_val_reps, unc_test_reps = split_reps(unc_reps, tng_ratio)
            tng_list += unc_tng_reps
            val_list += unc_val_reps
            test_list += unc_test_reps
            if verbose:
                tngc, valc, tstc = (
                    len(compl_tng_reps),
                    len(compl_val_reps),
                    len(compl_test_reps),
                )
                tngu, valu, tstu = (
                    len(unc_tng_reps),
                    len(unc_val_reps),
                    len(unc_test_reps),
                )
                print(
                    f"{sub} {level} completed: {tngc:02}/{valc:02}/{tstc:02}",
                    f"uncompleted: {tngu:02}/{valu:02}/{tstu:02}",
                )

    return tng_list, val_list, test_list


def main(args):
    categories = "all" if args.categories == ["all"] else args.categories
    tng_list, val_list, test_list = split_stratified_completed(
        args.data_path, categories, args.tng_ratio, args.verbose
    )
    split_dict = {"training": tng_list, "validation": val_list, "test": test_list}
    with open(args.out_file, "w") as out_file:
        json.dump(split_dict, out_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", dest="data_path", type=str, help="Path to the mario hdf5 file."
    )
    parser.add_argument("-v", dest="verbose", action="store_true")
    parser.add_argument(
        "-o", dest="out_file", type=str, help="Path of output json file."
    )
    parser.add_argument(
        "-r",
        dest="tng_ratio",
        type=float,
        help="Ratio of runs kept for training, e.g. 0.8 will keep 80 percents of "
        "runs in training, 10 percents in validation and 10 precents in test.",
        default=0.8,
    )
    parser.add_argument(
        "-c",
        dest="categories",
        type=str,
        nargs="*",
        default="all",
        help="Categories of level to use, default is 'all'.",
    )
    args = parser.parse_args()
    main(args)
