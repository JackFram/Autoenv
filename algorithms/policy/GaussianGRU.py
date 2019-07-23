import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

from algorithms.policy.GRUNetwork import GRUNetwork


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

        if feature_network is None:
            feature_dim = input_dim

        self.mean_network = GRUNetwork(
            input_dim=feature_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            gru_layer=gru_layer,
            output_nonlinearity=output_nonlinearity
        )

        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.fc_std.weight.fill_(np.log(init_std))

        # TODO: check if need to initialize bias

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.prev_actions = None
        self.prev_hiddens = None

    def forward(self, x, h=None):
        x, h = self.mean_network.forward(x, h)
        log_std = self.fc_std(h)
        return x, log_std, h

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


