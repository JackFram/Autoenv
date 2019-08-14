import torch
import torch.nn as nn
import numpy as np

from algorithms.policy.GRUNetwork import GRUNetwork
from algorithms.distribution.recurrent_diagonal_gaussian import RecurrentDiagonalGaussian
from algorithms.RL_Algorithm.optimizers.utils.math import normal_log_density
from algorithms.policy.GRUCell import GRUCell


class GaussianGRUPolicy(nn.Module):
    def __init__(self,
                 env_spec,
                 hidden_dim=32,
                 feature_network=None,
                 state_include_action=True,
                 gru_layer=GRUCell,
                 output_nonlinearity=None,
                 mode: int=0,
                 log_std=0,
                 cuda_enable=True):
        super().__init__()
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        if state_include_action:
            input_dim = obs_dim + action_dim
        else:
            input_dim = obs_dim

        # if feature_network is None:
        feature_dim = input_dim
        self._env_spec = env_spec

        self.mean_network = GRUNetwork(
            input_dim=feature_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            gru_layer=gru_layer,
            output_nonlinearity=output_nonlinearity
        )
        self.feature_network = feature_network

        # self.fc_std = nn.Linear(hidden_dim, action_dim).double()
        # self.fc_std.weight.data.fill_(np.log(init_std))
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

        # TODO: check if need to initialize bias

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.prev_actions = None
        self.prev_hiddens = None
        self.dist = RecurrentDiagonalGaussian(action_dim)

        self.state_include_action = state_include_action

        self.mode = mode
        self.cuda_enable = cuda_enable and torch.cuda.is_available()

        self.is_disc_action = False

    def forward(self, x, h=None):
        action_mean, h = self.mean_network.forward(x, h)
        # action_log_std = self.fc_std(h)
        action_log_std = self.action_log_std.expand_as(action_mean)
        return action_mean, action_log_std, h

    def load_param(self, param_path: str):
        self.load_state_dict(torch.load(param_path))

    def dist_info_sym(self, obs_var, state_info_vars):
        n_batches = np.array(obs_var).shape[0]
        n_steps = np.array(obs_var).shape[1]
        obs_var = torch.tensor(obs_var)
        obs_var = torch.reshape(obs_var, (n_batches, n_steps, -1))
        if self.state_include_action:
            prev_action_var = state_info_vars["prev_action"]
            all_input_var = torch.cat((obs_var, prev_action_var), dim=2)
        else:
            all_input_var = obs_var
        if self.feature_network is None:
            means, log_stds, _ = self.forward(all_input_var)
        else:
            flat_input_var = torch.reshape(all_input_var, (-1, self.input_dim))
            feature_batch = self.feature_network(flat_input_var)
            means, log_stds, _ = self.forward(feature_batch)
        return dict(mean=means, log_std=log_stds)

    def get_kl(self, x, actions, h=None):
        if self.state_include_action:
            prev_act = np.concatenate([np.zeros((actions.shape[0], 1, actions.shape[2])), actions], axis=1)[:, :-1, :]
            x = np.concatenate([x, prev_act], axis=-1)
        mean1, log_std1, std1 = self.forward(x, h)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        if self.state_include_action:
            prev_act = np.concatenate([np.zeros((actions.shape[0], 1, actions.shape[2])), actions], axis=1)[:, :-1, :]
            x = np.concatenate([x, prev_act], axis=-1)
        x = x.reshape((-1, self.input_dim))
        actions = actions.reshape((-1, self.action_dim))
        x = torch.tensor(x).cuda()
        actions = torch.tensor(actions).float().cuda()
        action_mean, action_log_std, hidden_vec = self.forward(x)
        action_log_std = action_log_std
        action_std = torch.exp(action_log_std)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x, actions):
        if self.state_include_action:
            prev_act = np.concatenate([np.zeros((actions.shape[0], 1, actions.shape[2])), actions], axis=1)[:, :-1, :]
            x = np.concatenate([x, prev_act], axis=-1)
        x = torch.tensor(x).reshape((-1, self.input_dim)).cuda()
        mean, action_log_std, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_dim))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))

        self.prev_actions[dones] = 0.
        if all(dones):
            self.prev_hiddens = None
        elif any(dones):
            self.prev_hiddens[dones] = None

    def get_action(self, observation):
        actions, agent_infos, _ = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        # mode: 0 stand for training, 1 for testing
        flat_obs = self.observation_space.flatten_n(observations)
        # self.prev_actions.shape = np.zeros([1,2], dtype=float)
        if self.state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([
                flat_obs,
                self.prev_actions
            ], axis=-1)
        else:
            all_input = flat_obs
        all_input = torch.tensor(all_input)
        if self.cuda_enable:
            all_input = all_input.cuda()
            if self.prev_hiddens is not None:
                self.prev_hiddens = self.prev_hiddens.cuda()
        means, log_stds, hidden_vec = self.forward(all_input, self.prev_hiddens)

        rnd = np.random.normal(size=means.shape)
        means = means.cpu().detach().numpy()
        log_stds = log_stds.cpu().detach().numpy()
        actions = rnd * np.exp(log_stds) + means
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        agent_info = dict(mean=means, log_std=log_stds)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        if self.mode == 1:
            return actions, agent_info, hidden_vec.cpu().detach().numpy()
        elif self.mode == 0:
            return actions, agent_info
        else:
            raise NotImplementedError

    def get_actions_with_prev(self, observations, prev_actions, prev_hiddens):
        # for getting back to hidden vector and action prediction before prediction
        if prev_actions is None or prev_hiddens is None:
            return self.get_actions(observations)
        flat_obs = self.observation_space.flatten_n(observations)
        # print(flat_obs.shape, prev_actions.shape)
        if self.state_include_action:
            h, w = flat_obs.shape
            all_input = np.concatenate([
                flat_obs,
                np.reshape(prev_actions, [h, 2])
                # np.zeros([h,2], dtype=float)
            ], axis=-1)
        else:
            all_input = flat_obs
        all_input = torch.tensor(all_input).cuda()
        if not torch.is_tensor(prev_hiddens):
            prev_hiddens = torch.tensor(prev_hiddens).float().cuda()
        means, log_stds, hidden_vec = self.forward(all_input, prev_hiddens)
        means = means.cpu().detach().numpy()
        log_stds = log_stds.cpu().detach().numpy()
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means

        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        agent_info = dict(mean=means, log_std=log_stds)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        if self.mode == 1:
            return actions, agent_info, hidden_vec.cpu().detach().numpy()
        elif self.mode == 0:
            return actions, agent_info
        else:
            raise NotImplementedError

    @property
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_specs(self):
        if self.state_include_action:
            return [
                ("prev_action", (self.action_dim,)),
            ]
        else:
            return []

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space




