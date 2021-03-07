import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PYG_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super(PYG_GCN, self).__init__()

        self.act_fn_in = nn.ReLU()
        self.act_fn_out = nn.ReLU()
        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, out_channels),
                                                 self.act_fn_in)
        self.gcn_layer = GCNConv(out_channels, out_channels)
        self.linear_cls_out = nn.Sequential(self.act_fn_out,
                                            nn.Linear(out_channels, num_class))

    def forward(self, x, edge_index, edge_weight=None):
        x = self.linear_transform_in(x)
        x = self.gcn_layer(x, edge_index, edge_weight)
        y_hat = self.linear_cls_out(x)
        return y_hat
