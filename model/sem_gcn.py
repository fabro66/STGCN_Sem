from __future__ import absolute_import
from functools import reduce

import torch.nn as nn
from model.sem_graph_conv import SemGraphConv
from model.graph_non_local import GraphNonLocal


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # x: (B, T, K, C) --> (B, C, T, K)
        x = self.gconv(x).permute(0, 3, 1, 2)
        # x: (B, C, T, K) --> (B, T, K, C)
        x = self.bn(x).permute(0, 2, 3, 1)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, group_size=1):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)

    def forward(self, x):
        # x: (B, T, N, C) --> (B*T, N, C)
        x_size = x.shape
        x = x.contiguous()
        x = x.view(-1, *x_size[2:])
        # x: (B*T, C, N)
        x = x.permute(0, 2, 1)

        x = self.non_local(x)

        # x: (B*T, C, N) --> (B*T, N, C)
        x = x.permute(0, 2, 1).contiguous()
        # x: (B*T, N, C) --> (B, T, N, C)
        x = x.view(*x_size)

        return x


class SemGCN(nn.Module):
    def __init__(self, adj, input_dim=128, hid_dim=128, coords_dim=3, num_layers=4, non_local=True, p_dropout=None):
        super(SemGCN, self).__init__()

        self.num_layers = num_layers

        _gconv_input = [_GraphConv(adj, input_dim, hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        if non_local:
            group_size = 1
            _gconv_input.append(_GraphNonLocal(hid_dim, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.gconv_output_1 = SemGraphConv(hid_dim, coords_dim, adj)
        self.gconv_output_end = SemGraphConv(hid_dim, coords_dim, adj)

    def forward(self, x):
        x = self.gconv_input(x)

        for i in range(self.num_layers):
            x = self.gconv_layers[2*i](x)
            x = self.gconv_layers[2*i + 1](x)

            if i == 2:
                out_1 = self.gconv_output_1(x)

        out_end = self.gconv_output_end(x)

        return out_1, out_end
