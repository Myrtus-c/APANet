import torch
from torch import nn
import torch.nn.functional as F

class AnomalyNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, dropout, theta):
        super(AnomalyNet, self).__init__()

        self.act_fn_in = nn.Tanh()
        self.act_fn_out = nn.ReLU()
        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, out_channels),
                                                 self.act_fn_in)
        # self.bernstein = Bernstein()
        self.adptive = AdaptiveFilter()
        self.high_agg = HighAggregation(out_channels, out_channels)
        self.linear_cls_out = nn.Sequential(self.act_fn_out,
                                            nn.Linear(out_channels, num_class))

        self.w = nn.Parameter(torch.randn(3, 1))

        self.params_filter = list(self.adptive.parameters())
        self.params_linear = list(self.linear_transform_in.parameters())
        self.params_linear.extend(list(self.linear_cls_out.parameters()))

        self.dropout = dropout

    def forward(self, x, L):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear_transform_in(x)
        low, high = self.adptive(x, L)
        # low, band, high = self.bernstein(x, L)
        delta = torch.norm((low - x), p=2, dim=1).reshape(-1, 1)
        high_delta = self.high_agg(delta, L)
        normal_delta = (high_delta - high_delta.min()) / (high_delta.max() - high_delta.min())
        # final_signals = torch.cat([low, band, high], dim=1)
        # final_signals = F.dropout(final_signals, self.dropout, training=self.training)
        # final_signals = self.w[0] * low + self.w[1] * band + self.w[2] * high
        final_signals = (1-normal_delta) * low + normal_delta * high
        y_hat = self.linear_cls_out(final_signals)
        return F.log_softmax(y_hat, dim=1)

class Bernstein(nn.Module):
    def __init__(self):
        super(Bernstein, self).__init__()
        self.delta = nn.Parameter(torch.zeros(1))

    def forward(self, x, L):
        Tx_0 = x
        Tx_1 = torch.spmm(L, Tx_0)
        Tx_2 = torch.spmm(L, Tx_1)

        low = Tx_2 + (-2. * self.delta - 2.) * Tx_1 + (self.delta + 1.)**2 * Tx_0
        band = 2. * (-Tx_2 + (2. * self.delta + 1.) * Tx_1 - (self.delta**2 + self.delta) * Tx_0)
        high = Tx_2 - 2. * self.delta * Tx_1 + self.delta**2 * Tx_0

        return low, band, high

class HighAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HighAggregation, self).__init__()
        # self.w = nn.Parameter(torch.randn(in_channels, out_channels))
        #
        # nn.init.kaiming_normal(self.w)

    def forward(self, x_high, L):
        Tx_0 = torch.spmm(L, x_high)
        return Tx_0

class AdaptiveFilter(nn.Module):
    def __init__(self):
        super(AdaptiveFilter, self).__init__()
        self.delta = nn.Parameter(torch.ones(1))
        self.a = nn.Parameter(torch.ones(1))

    def forward(self, x, L):
        Tx_0 = x
        Tx_1 = torch.spmm(L, x)
        Tx_2 = torch.spmm(L, Tx_1)
        Tx_3 = torch.spmm(L, Tx_2)

        low = Tx_3 + (-3. * self.delta - self.a) * Tx_2 + (3. * self.delta**2 + 2. * self.delta * self.a) * Tx_1 - (self.delta**3 + self.delta**2 * self.a) * Tx_0
        high = Tx_3 + (-3. * self.delta + self.a) * Tx_2 + (3. * self.delta**2 - 2. * self.delta * self.a) * Tx_1 + (self.delta**2 * self.a - self.delta**3) * Tx_0

        return low, high