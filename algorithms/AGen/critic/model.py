import torch.nn as nn
import torch


'''
Block of nn
'''


class Block(nn.Module):
    def __init__(self, input_size, hidden_layer_dims, activation_fn, drop_out_fn):
        super(Block, self).__init__()
        net = []
        input = input_size
        for hidden_size in hidden_layer_dims:
            net.append(nn.Linear(input, hidden_size))
            net.append(activation_fn)
            net.append(drop_out_fn)
            input = hidden_size
        self.block = nn.Sequential(*net)

    def forward(self, x):
        return self.block(x)


'''
Reward function approximation function(Neural Network)
'''


class ObservationActionMLP(nn.Module):
    def __init__(
            self,
            hidden_layer_dims,
            obs_size,
            act_size,
            output_dim=1,
            obs_hidden_layer_dims=list(),
            act_hidden_layer_dims=list(),
            activation_fn = nn.ReLU(inplace=False),
            dropout_keep_prob=1.,
            l2_reg=0.,
            return_features=False):
        super(ObservationActionMLP, self).__init__()
        self.output_dim = output_dim
        self.obs_hidden_layer_dims = obs_hidden_layer_dims
        self.act_hidden_layer_dims = act_hidden_layer_dims
        self.hidden_layer_dims = hidden_layer_dims
        self.activation_fn = activation_fn
        self.dropout_keep_prob = dropout_keep_prob
        self.return_features = return_features
        self.dropout = nn.Dropout(p=1 - dropout_keep_prob)
        self.l2_reg = l2_reg

        # build block for obs and action
        self.obs_block = Block(obs_size, obs_hidden_layer_dims, activation_fn, self.dropout)
        self.act_block = Block(act_size, act_hidden_layer_dims, activation_fn, self.dropout)
        feature_size = (self.obs_hidden_layer_dims[-1] if len(obs_hidden_layer_dims) != 0 else obs_size) \
                        + (self.act_hidden_layer_dims[-1] if len(act_hidden_layer_dims) != 0 else act_size)
        self.hidden_block = Block(feature_size, hidden_layer_dims, activation_fn, self.dropout)
        self.score_layer = nn.Linear(hidden_layer_dims[-1], output_dim)

    def forward(self, obs, act):
        if torch.cuda.is_available():
            obs = torch.tensor(obs).cuda().float()
            act = torch.tensor(act).cuda().float()
        else:
            obs = torch.tensor(obs).float()
            act = torch.tensor(act).float()
        obs_feature = self.obs_block.forward(obs)
        act_feature = self.act_block.forward(act)
        feature = torch.cat((obs_feature, act_feature), dim=1)
        feature = self.hidden_block.forward(feature)
        score = self.score_layer.forward(feature)
        return score

