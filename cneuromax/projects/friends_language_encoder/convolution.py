import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class AlignHrf(nn.Module):
    """Custom output layer"""

    def __init__(
        self: "AlignHrf",
        in_features: list[int],
        out_features: list[int],
    ) -> None:
        super().__init__()
        self.in_features, self.out_features = (
            in_features,
            out_features,
        )  # in_features is number of [embeddings x delay], out_features is embeddings  x voxels
        weight_matrix = np.random.rand(
            self.in_features[1],
            self.out_features[1],
        )
        weights = torch.tensor(weight_matrix)
        self.weights = nn.Parameter(
            weights,
        )  # nn.Parameter is a Tensor that's a module parameter.
        print(self.weights.size())
        bias_matrix = np.random.rand(self.in_features[0], 1)
        bias = torch.tensor(bias_matrix)
        self.bias = nn.Parameter(bias)
        # if bias:
        #     self.bias = nn.Parameter(bias)
        # else:
        #     self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self: "AlignHrf") -> None:
        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(
        self: "AlignHrf",
        features: Any,
        hrf_weight: list[float],
    ) -> torch.Tensor:
        """Models the delayed and dispersed nature of the HRF.

        in response to each word.

        Args:
            - features: The word feature matrix, features x word.
            - hrf_weight: Vector of weights derived from the HRF,
            adjusted for the sampling frequency of the fMRI data.

        Result:
            - summed hrf * word feature

        """
        w_times_matrix = np.random.rand(
            self.in_features[0],
            self.out_features[1],
        )
        w_times_x = torch.tensor(w_times_matrix)

        for word in range(
            1,
            features.shape[1],
        ):  # x is a word x features matrix
            w_times_x += (
                torch.matmul(features[word, :], self.weights)
                * hrf_weight[features.shape[1] - word]
            )

        return torch.add(w_times_x, self.bias)  # w times x + bweights


###### Initialize and use the model example let say we have 10 fmri data with same stimuli and training the model for predicting 11th fmri data


# Initialize the model
in_features = [100, 100]  # Assuming 100 words, 100-dimensional embeddings
out_features = [100, 50]  # Assuming outputs are 50 voxels
model = fMRIAlignment(in_features, out_features, model=None)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assume x_embeddings is a [100, 100] tensor for the 100 words with 100 features each
x_embeddings = torch.randn(
    100, 100
)  # Example word embeddings (constant across repetitions)
hrf_weights = torch.linspace(0, 1, steps=100)  # Example HRF weights

# Training data
# y_train: tensor of size [10, 100, 50] - 10 repetitions, each with responses over 50 voxels
y_train = torch.randn(
    10, 100, 50
)  # Example target fMRI data for 10 repetitions


### Training loop

epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for i in range(y_train.shape[0]):  # Loop over each dataset (repetition)
        optimizer.zero_grad()
        outputs = model(x_embeddings, hrf_weights)  # same embeddings each time
        loss = criterion(
            outputs, y_train[i]
        )  # different fMRI data per repetition
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / y_train.shape[0]}")


### Predict fMRI data

# Predict the fMRI data for the 11th repetition of the same words
predicted_fmri = model(x_embeddings, hrf_weights)
