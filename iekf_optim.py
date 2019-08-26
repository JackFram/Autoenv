import joblib
import numpy as np
import torch
from torch.optim import Optimizer


class IEKF(Optimizer):
    """iterative extended kalman filter optimizer.
    """

    def __init__(self, params, dim_out, p0=1e-2, eps=0, lbd=0.998, alpha=None, lbd_decay=False,
                 lbd_max_step=1000):

        if alpha is None:
            alpha = max(1 / lbd - 1, 0)
        self._check_format(dim_out, p0, eps, lbd, alpha, lbd_decay, lbd_max_step)
        defaults = dict(p0=p0, eps=eps, lbd=lbd, alpha=alpha,
                        lbd_decay=lbd_decay, lbd_max_step=lbd_max_step)
        super(IEKF, self).__init__(params, defaults)

        self.dim_out = dim_out
        with torch.no_grad():
            self._init_iekf_matrix()

    def _check_format(self, dim_out, p0, eps, lbd, alpha, lbd_decay, lbd_max_step):
        if not isinstance(dim_out, int) and dim_out > 0:
            raise ValueError("Invalid output dimension: {}".format(dim_out))
        if not 0.0 < p0:
            raise ValueError("Invalid initial P value: {}".format(p0))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 < lbd:
            raise ValueError("Invalid lambda parameter {}".format(lbd))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))
        if not isinstance(lbd_decay, int) and not isinstance(lbd_decay, bool):
            raise ValueError("Invalid lambda decay flag: {}".format(lbd_decay))
        if not isinstance(lbd_max_step, int):
            raise ValueError("Invalid max step for lambda decaying: {}".format(lbd_max_step))

    def _init_iekf_matrix(self):
        self.state['step'] = 0
        self.state['iekf_groups'] = []
        for group in self.param_groups:
            iekf_mat = []
            for p in group['params']:
                matrix = {}
                size = p.size()
                dim_w = 1
                for dim in size:
                    dim_w *= dim
                device = p.device
                matrix['P'] = group['p0'] * torch.eye(dim_w, dtype=torch.float, device=device, requires_grad=False)
                matrix['EPS'] = group['eps'] * torch.eye(dim_w, dtype=torch.float, device=device, requires_grad=False)
                matrix['lbd'] = group['lbd']
                matrix['H'] = None
                matrix['device'] = device
                iekf_mat.append(matrix)
            self.state['iekf_groups'].append(iekf_mat)

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        example:
            update_step=1
            coordinate_dim=2
            # predicted value
            y_hat = model(x)
            # true label: y
            y = y[:update_step*coordinate_dim].contiguous().view((-1, 1))
            y_hat = y_hat[:update_step*coordinate_dim].contiguous().view((-1, 1))
            err = (y - y_hat).detach()
            def closure(index=0):
                optimizer.zero_grad()
                retain = index < (update_step * coordinate_dim - 1)
                y_hat[index].backward(retain_graph=retain)
                return err
        """
        self.state['step'] += 1
        for y_ind in range(self.dim_out):
            err = closure(y_ind)
            for group_ind in range(len(self.param_groups)):
                group = self.param_groups[group_ind]
                iekf_mat = self.state['iekf_groups'][group_ind]
                for ii, w in enumerate(group['params']):
                    if w.grad is None:
                        continue
                    H_n = iekf_mat[ii]['H']
                    grad = w.grad.data.detach()
                    if len(w.size()) > 1:
                        grad = grad.transpose(1, 0)
                    grad = grad.contiguous().view((1, -1))
                    if y_ind == 0:
                        H_n = grad
                    else:
                        H_n = torch.cat([H_n, grad], dim=0)
                    self.state['iekf_groups'][group_ind][ii]['H'] = H_n

        err_T = err.transpose(0, 1)

        for group_ind in range(len(self.param_groups)):
            group = self.param_groups[group_ind]
            iekf_mat = self.state['iekf_groups'][group_ind]
            for ii, w in enumerate(group['params']):
                if w.grad is None:
                    continue

                lbd_n = iekf_mat[ii]['lbd']
                P_n = iekf_mat[ii]['P']
                EPS = iekf_mat[ii]['EPS']
                H_n = iekf_mat[ii]['H']
                H_n_T = H_n.transpose(0, 1)
                if group['lbd_decay']:
                    miu = 1.0 / min(self.state['step'], group['lbd_max_step'])
                    lbd_n = lbd_n + miu * (err_T.mm(err).flatten()[0] / self.dim_out - lbd_n)
                    self.state['iekf_groups'][group_ind][ii]['lbd'] = lbd_n
                R_n = lbd_n * torch.eye(self.dim_out, dtype=torch.float, device=iekf_mat[ii]['device'],
                                        requires_grad=False)

                g_n = R_n + H_n.mm(P_n).mm(H_n_T)
                g_n = g_n.inverse()

                K_n = P_n.mm(H_n_T).mm(g_n)
                delta_w = K_n.mm(err)
                if len(w.size()) > 1:
                    delta_w = delta_w.view((w.size(1), w.size(0))).transpose(1, 0)
                else:
                    delta_w = delta_w.view(w.size())

                new_P = (group['alpha'] + 1) * (P_n - K_n.mm(H_n).mm(P_n) + EPS)
                self.state['iekf_groups'][group_ind][ii]['P'] = new_P

                w.data.add_(delta_w)

        return err
