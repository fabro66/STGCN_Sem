from __future__ import absolute_import, division

import torch
from torch import nn


class _NonLocalBlock(nn.Module):
    def __init__(self, adj, in_channels, inter_channels=None, dimension=2):
        super(_NonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3]

        self.adj = adj
        self.concat = True
        self.dimension = dimension
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2)

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        assert self.inter_channels > 0

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        elif dimension == 1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        else:
            raise Exception('Error feature dimension.')

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
        )

        nn.init.kaiming_normal_(self.concat_project[0].weight)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)  # x: (b, c, N)/(b, c, h, w)

        # g_x: (b, N, c/2)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c/2, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c/2, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        # h: N, w: N
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.expand(-1, -1, -1, w)  # (b, c/2, N, N)
        phi_x = phi_x.expand(-1, -1, h, -1)

        # concat_feature: (b, c, N, N)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)  # (b, 1, N, N)
        b, _, h, w = f.size()
        e = self.leakyrelu(f.view(b, h, w))  # (b, N, N)

        zero_vec = -9e15*torch.ones_like(e)
        self.adj = self.adj.to(e.device)
        attention = torch.where(self.adj > 0, zero_vec, e)
        attention = self.softmax(attention)
        # y: (b, c, N)
        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        return y


class GraphNonLocal(_NonLocalBlock):
    def __init__(self, in_channels, inter_channels=None, dimension=1, bn_layer=True):
        super(GraphNonLocal, self).__init__(in_channels, inter_channels=inter_channels, dimension=1, bn_layer=bn_layer)
