import torch
from torch import nn
import torch.nn.functional as F
import numpy  as np
from einops import repeat

class NodeBernNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, attn='Concat', filter_num=2, K=3, softmax=False):
        super(NodeBernNet, self).__init__()

        self.K = K

        self.filter_num = filter_num

        if self.K == 4:
            theta_list = [[.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0]]

        elif self.K == 3:
            theta_list = [[0., 0., 0.0, 0.0,],
                      [0.0, 0.0, 0., 0.,]]

        self.filters = nn.ModuleList([BernsteinLayer(theta_list[i], self.K, out_channels, softmax_theta=softmax) for i in range(filter_num)])


        self.act_fn = nn.ReLU()
        self.tanh = nn.Tanh()
        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, out_channels),
                                                 self.act_fn,)


        self.attn = attn
        self.linear_cls_out = nn.Sequential(nn.Linear(out_channels, out_channels),
                                            self.act_fn,
                                            nn.Linear(out_channels, out_channels // 2),
                                            self.act_fn,
                                            nn.Linear(out_channels // 2, num_class)
                                            )

        self.params_filter = list(self.filters.parameters())
        self.params_linear = list(self.linear_transform_in.parameters())
        self.params_linear.extend(list(self.linear_cls_out.parameters()))


        if self.attn == 'Bili':
            # Bilinear Attention
            self.Wb = nn.Linear(out_channels, out_channels, bias=True)
            self.Wx = nn.Linear(out_channels, out_channels, bias=True)
            self.params_attn = list(self.Wb.parameters())
            self.params_attn.extend(self.Wx.parameters())

        elif self.attn == 'Concat':
            # Concat Attention
            self.Wb = nn.Linear(out_channels, out_channels, bias=True)
            self.Wx = nn.Linear(out_channels, out_channels, bias=True)
            self.vc = nn.Parameter(torch.Tensor(out_channels, 1))
            nn.init.kaiming_normal_(self.vc)

            self.params_attn = list(self.Wb.parameters())
            self.params_attn.extend(self.Wx.parameters())




    def forward(self, x, L, exp=False, label=None):
        """
        :param x:
        :param L:
        :param exp:
        :param label:   [B * 1]
        :return:
        """
        x = self.linear_transform_in(x)
        if self.training:
            anomaly_train, normal_train = label

        poly_list = []

        for i, filter_ in enumerate(self.filters):
            poly = filter_(x, L)
            poly_list.append(poly)

        poly = torch.stack(poly_list, dim=1)

        if self.attn == 'Bili':
            poly_proj = self.Wb(poly)
            x_proj = self.Wx(x).unsqueeze(-1)
            score = torch.bmm(poly_proj, x_proj)

        elif self.attn == 'Concat':
            b = x.shape[0]
            vc = repeat(self.vc, 'c o -> b c o', b=b)
            x_repeat = repeat(x, 'b c -> b f c', f=self.filter_num)
            poly_proj = self.Wb(poly)
            x_proj = self.Wx(x_repeat)
            score = torch.bmm(self.tanh(poly_proj + x_proj), vc)


        # hard_logit = F.gumbel_softmax(score, hard=True, dim=1, tau=3.0)
        hard_logit = F.softmax(score, dim=1)

        res = poly[:, 0, :] * hard_logit[:, 0]
        for i in range(1, self.filter_num):
            res += poly[:, i, :] * hard_logit[:, i]

        y_hat = self.linear_cls_out(res)

        if self.training:
            normal_bias = torch.mean(hard_logit[normal_train][:, 0] - hard_logit[normal_train][:, 1])
            anomaly_bias = torch.mean(hard_logit[anomaly_train][:, 1] - hard_logit[anomaly_train][:, 0])
            bias_loss = torch.exp(normal_bias + anomaly_bias)
            return y_hat, bias_loss

        if exp is True:
            return y_hat, hard_logit
        return y_hat



class BernsteinLayer(nn.Module):
    def __init__(self, theta_list, K, out_channels, softmax_theta=True):
        super(BernsteinLayer, self).__init__()
        assert len(theta_list) == K + 1
        self.K = K

        self.theta0 = nn.Parameter(torch.FloatTensor([theta_list[0]]))
        self.theta1 = nn.Parameter(torch.FloatTensor([theta_list[1]]))
        self.theta2 = nn.Parameter(torch.FloatTensor([theta_list[2]]))
        self.theta3 = nn.Parameter(torch.FloatTensor([theta_list[3]]))
        if self.K == 4:
            self.theta4 = nn.Parameter(torch.FloatTensor([theta_list[4]]))

        self.softmax_theta = softmax_theta
        if self.softmax_theta is True:
            self.softmax = nn.Softmax(dim=1)

        # self.linear = nn.Linear(out_channels, out_channels)

    def forward(self, x, L):
        Tx0 = x
        Tx1 = torch.spmm(L, x)
        Tx2 = torch.spmm(L, Tx1)
        Tx3 = torch.spmm(L, Tx2)

        if self.K == 3:
            Bx03 = Tx0 - 3 * Tx1 + 3 * Tx2 - Tx3
            Bx13 = 3 * Tx1 - 6 * Tx2 + 3 * Tx3
            Bx23 = 3 * Tx2 - 3 * Tx3
            Bx33 = Tx3

        elif self.K == 4:
            Tx4 = torch.spmm(L, Tx3)
            Bx04 = Tx0 - 4. * Tx1 + 6. * Tx2 - 4. * Tx3 + Tx4
            Bx14 = 4. * Tx1 - 12. * Tx2 + 12. * Tx3 - 4. * Tx4
            Bx24 = 6. * Tx2 - 12. * Tx3 + 6. * Tx4
            Bx34 = 4. * Tx3 - 4. * Tx4
            Bx44 = Tx4

        if self.K == 3:
            # Softmax weight
            if self.softmax_theta:
                theta_list = self.softmax(torch.stack([self.theta0, self.theta1, self.theta2, self.theta3], dim=1))[0]
                poly = theta_list[0] * Bx03
                poly += theta_list[1] * Bx13
                poly += theta_list[2] * Bx23
                poly += theta_list[3] * Bx33

            else:
                poly = self.theta0 * Bx03
                poly += self.theta1 * Bx13
                poly += self.theta2 * Bx23
                poly += self.theta3 * Bx33


        elif self.K == 4:
            if self.softmax_theta:
                theta_list = self.softmax(torch.stack([self.theta0, self.theta1, self.theta2, self.theta3, self.theta4], dim=1))[0]
                poly = theta_list[0] * Bx04
                poly += theta_list[1] * Bx14
                poly += theta_list[2] * Bx24
                poly += theta_list[3] * Bx34
                poly += theta_list[4] * Bx44


            poly = self.theta0 * Bx04
            poly += self.theta1 * Bx14
            poly += self.theta2 * Bx24
            poly += self.theta3 * Bx34
            poly += self.theta4 * Bx44

        return poly


if __name__ == '__main__':
    b = BernsteinLayer([1.0, 2.0, 3.0], 3)