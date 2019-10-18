from __future__ import absolute_import, division

import torch
from torch import nn


class TemNonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, num_joints_out=17, bn_layer=False):
        super(TemNonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.num_joints_outs = num_joints_out

        self.in_channels = in_channels
        self.inter_channels = inter_channels

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
            nn.Conv3d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        # self.concat_project = nn.Parameter(torch.zeros(size=(num_joints_out, num_joints_out, 1), dtype=torch.float))
        # nn.init.xavier_uniform_(self.concat_project.data, gain=1.414)
        # self.relu = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.concat_project[0].weight)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal_(self.W[0].weight)
            nn.init.constant_(self.W[0].bias, 0)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        #  x: (B, C, T, K)

        g_x = self.g(x)
        g_x = g_x.permute(0, 2, 3, 1)   # g_x: (B, T, K, C//2)

        # theta_x = (B, C//2, T, K, 1)
        theta_x = self.theta(x).contiguous()
        theta_x = theta_x.unsqueeze(dim=-1)
        # phi_x = (B, C//2, T, 1, K)
        phi_x = self.phi(x).contiguous()
        phi_x = phi_x.unsqueeze(dim=-2)

        # h: K, w: K
        h = theta_x.size(3)
        w = phi_x.size(4)
        theta_x = theta_x.expand(-1, -1, -1, -1, w)  # (B, C//2, T, K, K)
        phi_x = phi_x.expand(-1, -1, -1, h, -1)  # (B, C//2, T, K, K)

        # concat_feature: (B, C, T, K, K)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)  # (B, 1, T, K, K)
        b, _, t, k, k = f.size()
        f = f.view(b, t, k, k)  # (B, T, K, K)

        N = f.size(-1)
        f_div_C = f / N

        # y: (B, T, K, C//2)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 3, 1, 2).contiguous()
        W_y = self.W(y)
        z = W_y + x

        return z


# class TemNonLocalBlock(nn.Module):
#     def __init__(self, in_channels, num_att, dimension=2, num_joints=17, bn_layer=True):
#         super().__init__()
