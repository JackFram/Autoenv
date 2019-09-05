import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch

from algorithms.policy.MLP import MLP


class GaussianMLP(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            mean_network=None,
            optimizer=None,
            hidden_size=(32, 32),
            step_size=0.01,
            init_std=1.0,
            normalize_inputs=True,
            normalize_outputs=True,
            subsample_factor=1.0,
            max_itr=20
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

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mean_network = mean_network
        self.lr = step_size
        self.init_std = init_std
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.subsample_factor = subsample_factor
        if optimizer is None:
            optimizer = optim.RMSprop(mean_network.parameters(), lr=self.lr)
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()
        self.max_itr = max_itr

    def forward(self, x):
        if self.normalize_inputs:
            x = (x - x.mean(dim=0))/(x.std(dim=0)+1e-4)
        if torch.cuda.is_available():
            x = x.cuda()
        mean = self.mean_network(x)
        if self.normalize_outputs:
            mean = (mean - mean.mean(dim=0))/(mean.std(dim=0)+1e-4)
        mean = mean.double()
        return mean

    def fit(self, xs, ys):
        if torch.cuda.is_available():
            xs = torch.tensor(xs).double().cuda()
            ys = torch.tensor(ys).double().cuda()
        else:
            xs = torch.tensor(xs).double()
            ys = torch.tensor(ys).double()
        if self.subsample_factor < 1:
            num_samples_tot = xs.shape[0]
            idx = np.random.randint(0, num_samples_tot, int(num_samples_tot * self._subsample_factor))
            xs, ys = xs[idx], ys[idx]
        if self.normalize_outputs:
            ys_mean = ys.mean(dim=0)
            ys_std = ys.std(dim=0)
            ys = (ys - ys_mean)/(ys_std+1e-4)
        for itr in range(self.max_itr):
            # print("fitting xs: ", xs)
            output = self.forward(xs)
            # print("output: ", output)
            loss = self.criterion(output, ys)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, xs):
        xs = torch.tensor(xs).double()
        return self.forward(xs).cpu().detach().numpy()


class GaussianMLPBaseline(object):
    def __init__(
            self,
            env_spec,
            subsample_factor=1,
            num_seq_inputs=1,
            regressor_args=None
    ):
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianMLP(
            input_dim=env_spec.observation_space.flat_dim,
            output_dim=1,
            **regressor_args
        ).double()

    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    def predict(self, path):
            return self._regressor.predict(path["observations"]).flatten()

    def parameters(self):
        return self._regressor.parameters()

    def set_cuda(self):
        self._regressor.cuda()







