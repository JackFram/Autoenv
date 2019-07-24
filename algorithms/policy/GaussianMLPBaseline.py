import numpy as np
import torch.nn as nn
import torch.optim as optim

from algorithms.policy.MLP import MLP


class GaussianMLP(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            mean_network=None,
            hidden_size=(32, 32),
            step_size=0.01,
            init_std=1.0,
            normalize_inputs=True,
            normalize_outputs=True,
    ):
        """
        :param input_dim:
        :param output_dim:
        :param mean_network:
        :param hidden_size:
        :param step_size:
        :param init_std:
        :param normalize_inputs:
        :param normalize_outputs:
        """
        super(GaussianMLP, self).__init__()
        if mean_network is None:
            mean_network = MLP(input_size=input_dim, hidden_size=hidden_size, output_size=output_dim)
        self.fc_std = nn.Linear(input_dim, output_dim)
        self.fc_std.weight.fill_(np.log(init_std))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mean_network = mean_network
        self.lr = step_size
        self.init_std = init_std
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs

    def forward(self, x):
        if self.normalize_inputs:
            x = (x - x.mean(axis=0))/x.std(axis=0)
        mean = self.mean_network(x)
        std = self.fc_std(x)
        if self.normalize_outputs:
            mean = (mean - mean.mean(axis=0))/mean.std(axis=0)
            std = (std - np.log(mean.std(axis=0)))
        return mean, std
