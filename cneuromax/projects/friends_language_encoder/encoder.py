"""."""

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def accuracy_2v2(
    vector_1: tuple[list[float], list[float]],
    vector_2: tuple[list[float], list[float]],
) -> float:
    """Estimates 2v2 correlation."""
    vector_len = len(vector_1)
    # gives the combination of distinct pairs can be formed from the list
    # of items in the vectors.
    n_choose_2 = vector_len * (vector_len - 1) / 2
    sum_of_accuracy = 0

    for i in range(vector_len - 1):
        for j in range(i + 1, vector_len):
            if (
                cosine(vector_1[i], vector_2[i])
                + cosine(vector_1[j], vector_2[j])
            ) < (
                cosine(vector_1[i], vector_1[j])
                + cosine(vector_1[j], vector_2[i])
            ):
                sum_of_accuracy += 1

    return sum_of_accuracy / n_choose_2


# Pearson correlation
def accuracy_pc(
    vector_1: tuple[list[float], list[float]],
    vector_2: tuple[list[float], list[float]],
) -> float:
    """."""
    n = len(vector_1)
    total = 0

    for i in range(n):
        total += np.corrcoef(vector_1[i], vector_2[i])

    return total / n


class BrainEncoderLayer:
    """."""

    def __init__(
        self: "BrainEncoderLayer",
        fmridata: list[float],
        features: list[float],
        k: int,
    ) -> None:
        """."""
        self.fmridata = fmridata
        self.features = features
        self.k = k
        self.kfold = KFold(n_splits=k, shuffle=False, random_state=None)

    def train(self: "BrainEncoderLayer") -> dict[str, list[float]]:
        """."""
        eval_pearson = {}
        eval_2v2 = {}
        self.accuracy = {}
        for train_ids, test_ids in self.kfold.split(self.kfold):
            train_features, train_fmri = (
                self.features[train_ids],
                self.fmridata[train_ids],
            )
            test_features, test_fmri = (
                self.features[train_ids],
                self.fmridata[test_ids],
            )

            model = Ridge(alpha=1)
            model.fit(train_features, train_fmri)

            predicted = model.predict(test_features)

            eval_pearson += accuracy_pc(test_fmri, predicted)
            eval_2v2 += accuracy_2v2(test_fmri, predicted)
        self.accuracy["pc"] = eval_pearson[0][1] / self.k
        self.accuracy["2vector_2"] = eval_2v2 / self.k
        return self.accuracy
