import math

import numpy as np
import torch
import torch.nn as nn


class fMRIAlignment(nn.Module):
    """Custom output layer"""

    def __init__(self, in_features, out_features):
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

    def reset_parameters(self) -> None:
        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x, hrf_weight):
        w_times_matrix = np.random.rand(
            self.in_features[0], self.out_features[1]
        )
        w_times_x = torch.tensor(w_times_matrix)

        for word in range(1, x.shape[1]):  # x is a word x features matrix
            w_times_x += (
                torch.matmul(x[word, :], self.weights)
                * hrf_weight[x.shape[1] - word]
            )

        return torch.add(w_times_x, self.bias)  # w times x + bweights
