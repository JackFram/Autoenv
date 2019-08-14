import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        A gated recurrent unit implements the following update mechanism:
        Reset gate:        r(t) = f_r(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
        Update gate:       u(t) = f_u(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
        Cell gate:         c(t) = f_c(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
        New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u_t * c(t)
        Note that the reset, update, and cell vectors must have the same dimension as the hidden state
        """

        super(GRUCell, self).__init__()
        # Weights for the reset gate
        self.W_xr = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.Tensor(hidden_size, 1))
        # Weights for the update gate
        self.W_xu = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hu = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_u = nn.Parameter(torch.Tensor(hidden_size, 1))
        # Weights for the cell gate
        self.W_xc = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size, 1))

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gate_nonlinearity = nn.Sigmoid()
        self.nonlinearity = nn.Tanh()

    def forward(self, x, h=None):
        x = x.float()
        if h is None:
            h = x.new_zeros(x.size(0), self.hidden_size, requires_grad=False)
        W_x_ruc = torch.cat([self.W_xr, self.W_xu, self.W_xc], dim=1)
        W_h_ruc = torch.cat([self.W_hr, self.W_hu, self.W_hc], dim=1)
        b_ruc = torch.cat([self.b_r, self.b_u, self.b_c], dim=1)
        if torch.cuda.is_available():
            xb_ruc = torch.matmul(x, W_x_ruc.cuda()) + torch.reshape(b_ruc.cuda(), (1, -1))
            h_ruc = torch.matmul(h, W_h_ruc.cuda())
        else:
            xb_ruc = torch.matmul(x, W_x_ruc) + torch.reshape(b_ruc, (1, -1))
            h_ruc = torch.matmul(h, W_h_ruc)
        xb_r, xb_u, xb_c = torch.split(dim=1, split_size_or_sections=int(xb_ruc.shape[1]/3), tensor=xb_ruc)
        h_r, h_u, h_c = torch.split(dim=1, split_size_or_sections=int(h_ruc.shape[1]/3), tensor=h_ruc)
        r = self.gate_nonlinearity(xb_r + h_r)
        u = self.gate_nonlinearity(xb_u + h_u)
        c = self.nonlinearity(xb_c + r * h_c)
        h = (1 - u) * h + u * c
        return h
