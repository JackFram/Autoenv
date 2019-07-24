import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

from algorithms.policy.GRUNetwork import GRUNetwork
from algorithms.distribution.recurrent_diagonal_gaussian import RecurrentDiagonalGaussian


class GaussianGRUPolicy(nn.Module):
    def __init__(self,
                 env_spec,
                 hidden_dim=32,
                 feature_network=None,
                 state_include_action=True,
                 gru_layer=nn.GRUCell,
                 init_std=1.0,
                 output_nonlinearity=None):
        super(GaussianGRUPolicy, self).__init__()
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        if state_include_action:
            input_dim = obs_dim + action_dim
        else:
            input_dim = obs_dim

        # if feature_network is None:
        feature_dim = input_dim

        self.mean_network = GRUNetwork(
            input_dim=feature_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            gru_layer=gru_layer,
            output_nonlinearity=output_nonlinearity
        )
        self.feature_network = feature_network

        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.fc_std.weight.fill_(np.log(init_std))

        # TODO: check if need to initialize bias

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.prev_actions = None
        self.prev_hiddens = None
        self.dist = RecurrentDiagonalGaussian(action_dim)

    def forward(self, x, h=None):
        x, h = self.mean_network.forward(x, h)
        log_std = self.fc_std(h)
        return x, log_std, h

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

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_space.flat_dim))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))

        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self.mean_network.hid_init_param.eval()

    def get_action(self, observation):
        actions, agent_infos, _ = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
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

        means, log_stds, hidden_vec = self.forward(all_input, self.prev_hiddens)

        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        agent_info = dict(mean=means, log_std=log_stds)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        return actions, agent_info, hidden_vec

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

        means, log_stds, hidden_vec = self.forward(all_input, prev_hiddens)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means

        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        agent_info = dict(mean=means, log_std=log_stds)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        return actions, agent_info, hidden_vec

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




