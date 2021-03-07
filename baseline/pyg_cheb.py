import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class PYG_ChebNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, K):
        super(PYG_ChebNet, self).__init__()

        self.act_fn_in = nn.Tanh()
        self.act_fn_out = nn.ReLU()
        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, out_channels),
                                                 self.act_fn_in)
        self.cheb_layer = ChebConv(out_channels, out_channels, K=K)
        self.linear_cls_out = nn.Sequential(self.act_fn_out,
                                            nn.Linear(out_channels, num_class))

    def forward(self, x, edge_index, edge_weight=None):
        x = self.linear_transform_in(x)
        x = self.cheb_layer(x, edge_index, edge_weight)
        y_hat = self.linear_cls_out(x)
        return y_hat