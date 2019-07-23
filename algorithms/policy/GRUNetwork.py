import torch.nn as nn


class GRUNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, gru_layer=nn.GRUCell,
                 output_nonlinearity=None):
        super(GRUNetwork, self).__init__()
        self.gru = gru_layer(input_dim=input_dim, hidden_size=hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.output_activation_fn = output_nonlinearity

    def forward(self, x, h=None):
        if h is not None:
            h = self.gru(x, h)
        else:
            h = self.gru(x)
        x = self.fc(x)
        if self.output_activation_fn is not None:
            x = self.output_activation_fn(x)
        return x, h
