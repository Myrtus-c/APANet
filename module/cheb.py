import torch
from torch import nn
import torch.nn.functional as F

class ChebNet(nn.Module):
    def __init__(self, in_channels, K, out_channels, num_class, dropout, theta_linear, init_t=0.5):
        super(ChebNet, self).__init__()
        assert K >= 2, "Need at least order 2 Chebyshev."
        self.filters = nn.ModuleList()
        for _ in range(K):
            self.filters.append(ChebLayer(init_t))

        self.act_fn = nn.ReLU()
        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, out_channels),
                                                 self.act_fn,)
        self.linear_cls_out = nn.Sequential(nn.Linear(out_channels, out_channels//2),
                                            self.act_fn,
                                            nn.Linear(out_channels//2, num_class))

        self.params_filter = list(self.filters.parameters())
        self.params_linear = list(self.linear_transform_in.parameters())
        self.params_linear.extend(list(self.linear_cls_out.parameters()))

        # add theta linear transform
        self.theta_linear = theta_linear
        if self.theta_linear:
            self.weight = nn.Parameter(torch.zeros(K, out_channels, out_channels))
            nn.init.kaiming_normal_(self.weight)

        self.dropout = dropout
        self.K = K

    def forward(self, x, L):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear_transform_in(x)
        T_0, poly = self.filters[0](x, None)
        T_1, term = self.filters[1](T_0, None, L)

        if self.theta_linear:
            poly = torch.matmul(poly, self.weight[0])
            term = torch.matmul(term, self.weight[1])

        poly += term

        prevs = [T_0, T_1]
        for i, filter_ in enumerate(self.filters[2:]):
            T_i, term = filter_(prevs[0], prevs[1], L)

            if self.theta_linear:
                term = torch.matmul(term, self.weight[i + 2])

            prevs[1] = prevs[0]
            prevs[0] = T_i
            poly += term

        poly = F.dropout(poly, self.dropout, training=self.training)
        y_hat = self.linear_cls_out(poly)
        return y_hat



class ChebLayer(nn.Module):
    def __init__(self, theta):
        super(ChebLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor([theta]))

    def forward(self, T_n_1, T_n_2, M=None):
        if M is not None and T_n_2 is not None:
            H_l = 2 * torch.spmm(M, T_n_1)
            H_l = H_l - T_n_2
        elif M is not None and T_n_2 is None:
            H_l = torch.spmm(M, T_n_1)
        else:
            H_l = T_n_1
        return H_l, self.theta * H_l