import torch
import math
import torch.nn as nn
from model.sem_graph_conv import SemGraphConv
from model.sem_ch_graph_conv import SemCHGraphConv
from model.graph_attention_network import _NonLocalBlock
# from model.temporal_non_local import TemNonLocalBlock


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, dropout=None):
        super(_GraphConv, self).__init__()

        adj_2 = adj.matrix_power(2)
        adj_3 = adj.matrix_power(3)

        self.gconv_1 = SemCHGraphConv(input_dim, output_dim, adj)
        self.bn_1 = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.gconv_2 = SemCHGraphConv(input_dim, output_dim, adj_2)
        self.bn_2 = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.gconv_3 = SemCHGraphConv(input_dim, output_dim, adj_3)
        self.bn_3 = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.relu = nn.ReLU()

        self.cat_conv = nn.Conv2d(3*output_dim, output_dim, 1)
        self.cat_bn = nn.BatchNorm2d(output_dim, momentum=0.1)

        nn.init.constant_(self.cat_conv.bias, 0)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # x: (B, T, K, C)
        x_ = self.gconv_1(x)
        y_ = self.gconv_2(x)
        z_ = self.gconv_3(x)

        # x: (B, T, K, C) --> (B, C, T, K)
        x = x_.permute(0, 3, 1, 2)
        y = y_.permute(0, 3, 1, 2)
        z = z_.permute(0, 3, 1, 2)
        x = self.bn_1(x)
        y = self.bn_2(y)
        z = self.bn_3(z)

        if self.dropout is not None:
            x = self.dropout(self.relu(x))
            y = self.dropout(self.relu(y))
            z = self.dropout(self.relu(z))
        else:
            x = self.relu(x)
            y = self.relu(y)
            z = self.relu(z)

        output = torch.cat((x, y, z), dim=1)
        output = self.cat_bn(self.cat_conv(output))

        if self.dropout is not None:
            output = self.dropout(self.relu(output))
        else:
            output = self.relu(output)
        output = output.permute(0, 2, 3, 1)

        return output


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout):
        super(_ResGraphConv, self).__init__()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None
        hid_dim = output_dim
        self.relu = nn.ReLU(inplace=True)

        # self.conv_1 = nn.Conv2d(input_dim, hid_dim, 1, bias=False)
        # self.bn_1 = nn.BatchNorm2d(hid_dim, momentum=0.1)
        # self.conv_2 = nn.Conv2d(hid_dim, output_dim, 1, bias=False)
        # self.bn_2 = nn.BatchNorm2d(output_dim, momentum=0.1)

        self.gconv1 = _GraphConv(adj, hid_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, hid_dim, p_dropout)

    def forward(self, x):
        residual = x
        # x = self.relu(self.bn_1(self.conv_1(x)))
        # if self.dropout is not None:
        #     x = self.dropout(x)

        # x: (B, C, T, K) --> (B, T, K, C)
        x = x.permute(0, 2, 3, 1)
        x = self.gconv1(x)
        x = self.gconv2(x)

        # x: (B, T, K, C) --> (B, C, T, K)
        x = x.permute(0, 3, 1, 2)
        # x = self.relu(self.bn_2(self.conv_2(x)))
        # if self.dropout is not None:
        #     x = self.dropout(x)

        x = x + residual

        return x


class _GraphNonLocal(nn.Module):
    def __init__(self, adj, in_channels, inter_channels, dim=2):
        super(_GraphNonLocal, self).__init__()

        self.num_non_local = in_channels // inter_channels

        attentions = [_NonLocalBlock(adj, in_channels, inter_channels, dimension=dim) for _ in range(self.num_non_local)]
        self.attentions = nn.ModuleList(attentions)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.1)
        # for i, attention in enumerate(self.attentions):
        #     self.add_module("attention_{}".format(i), attention)

        # self.W = nn.Parameter(torch.zeros(size=(in_channels, in_channels), dtype=torch.float))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x):
        # x: (B, T, K, C) --> (B*T, K, C)
        x_size = x.shape
        x = x.contiguous()
        x = x.view(-1, *x_size[2:])
        # x: (B*T, C, K)
        x = x.permute(0, 2, 1)

        x = torch.cat([self.attentions[i](x) for i in range(len(self.attentions))], dim=1)

        # x: (B*T, C, K) --> (B*T, K, C)
        x = x.permute(0, 2, 1).contiguous()

        # x = torch.matmul(x, self.W)
        # x: (B*T, K, C) --> (B, T, K, C)
        x = x.view(*x_size)

        # x: (B, T, K, C) --> (B, C, T, K)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.bn(x))

        # x: (B, C, T, K) --> (B, T, K, C)
        x = x.permute(0, 2, 3, 1)

        return x


class GraphConvNonLocal(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout):
        super(GraphConvNonLocal, self).__init__()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None
        hid_dim = output_dim
        self.relu = nn.ReLU(inplace=True)
        self.adj = adj.matrix_power(3)

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

        self.non_local = _GraphNonLocal(adj, input_dim, input_dim//4, dim=1)
        self.cat_conv = nn.Conv2d(2*output_dim, output_dim, 1)
        self.cat_bn = nn.BatchNorm2d(output_dim, momentum=0.1)

        # nn.init.constant_(self.cat_conv.bias, 0)

    def forward(self, x):
        residual = x

        # x: (B, C, T, K) --> (B, T, K, C)
        x = x.permute(0, 2, 3, 1)
        x_ = self.gconv1(x)
        x_ = self.gconv2(x_)

        y_ = self.non_local(x)
        x = torch.cat((x_, y_), dim=-1)

        # x: (B, T, K, C) --> (B, C, T, K)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.cat_bn(self.cat_conv(x)))
        x = residual + x
        return x


class SpatialTemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, adj, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm2d(channels, momentum=0.1)
        self.shrink = nn.Conv2d(channels, 3, 1)

        nn.init.constant_(self.shrink.bias, 0)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        """
        X: (B, C, T, K)
            B: batchsize
            T: Temporal
            K: The number of keypoints
            C: The feature dimension of keypoints
        """

        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        # X: (B, T, K, C)
        x = self._forward_blocks(x)
        x = self.shrink(x)

        # x: (B, C, T, K) --> (B, T, K, C)
        x = x.permute(0, 2, 3, 1)

        return x


class SpatialTemporalModel(SpatialTemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, adj, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=64, dense=False):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(adj, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv2d(in_features, channels, (filter_widths[0], 1), bias=False)

        layers_conv = []
        layers_graph_conv = []
        # layers_non_local = []
        layers_bn = []

        layers_graph_conv.append(GraphConvNonLocal(adj, channels, channels, p_dropout=None))
        # layers_non_local.append(_NonLocalBlock(channels, channels // 2, dimension=2, bn_layer=True))

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            layers_conv.append(nn.Conv2d(channels, channels, (filter_widths[i], 1) if not dense else (2 * self.pad[-1] + 1, 1),
                                         dilation=(next_dilation, 1) if not dense else (1, 1), bias=False))
            layers_bn.append(nn.BatchNorm2d(channels, momentum=0.1))
            layers_conv.append(nn.Conv2d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm2d(channels, momentum=0.1))

            layers_graph_conv.append(GraphConvNonLocal(adj, channels, channels, p_dropout=None))
            # layers_non_local.append(_NonLocalBlock(channels, channels // 2, dimension=2, bn_layer=True))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_graph_conv = nn.ModuleList(layers_graph_conv)
        # self.layers_non_local = nn.ModuleList(layers_non_local)

    def _forward_blocks(self, x):

        # x: (B, T, K, C) --> (B, C, T, K)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.expand_bn(self.expand_conv(x)))
        # x = self.layers_non_local[0](x)
        x = self.layers_graph_conv[0](x)

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]

            # x: (B, C, T, K)
            x = self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x)))
            x = res + self.relu(self.layers_bn[2*i+1](self.layers_conv[2*i+1](x)))

            # x = self.layers_non_local[i+1](x)
            x = self.layers_graph_conv[i+1](x)

        return x


class SpatialTemporalModelOptimized1f(SpatialTemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, adj, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=64):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(adj, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv2d(in_features, channels, (filter_widths[0], 1), stride=(filter_widths[0], 1), bias=False)

        layers_conv = []
        layers_graph_conv = []
        # layers_non_local = []
        layers_bn = []

        layers_graph_conv.append(GraphConvNonLocal(adj, channels, channels, p_dropout=None))
        # layers_non_local.append(_NonLocalBlock(channels, channels//2, dimension=2, bn_layer=True))

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            layers_conv.append(nn.Conv2d(channels, channels, (filter_widths[i], 1), stride=(filter_widths[i], 1), bias=False))
            layers_bn.append(nn.BatchNorm2d(channels, momentum=0.1))
            layers_conv.append(nn.Conv2d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm2d(channels, momentum=0.1))

            layers_graph_conv.append(GraphConvNonLocal(adj, channels, channels, p_dropout=None))
            # layers_non_local.append(_NonLocalBlock(channels, channels // 2, dimension=2, bn_layer=True))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_graph_conv = nn.ModuleList(layers_graph_conv)
        # self.layers_non_local = nn.ModuleList(layers_non_local)

    def _forward_blocks(self, x):
        # x: (B, T, K, C) --> (B, C, T, K)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.expand_bn(self.expand_conv(x)))
        # x = self.layers_non_local[0](x)
        x = self.layers_graph_conv[0](x)

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]

            # x: (B, C, T, K)
            x = self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x)))
            x = res + self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x)))

            # x = self.layers_non_local[i+1](x)
            x = self.layers_graph_conv[i+1](x)

        return x


if __name__ == "__main__":
    import torch
    import numpy as np
    from common.skeleton import Skeleton
    from common.graph_utils import adj_mx_from_skeleton

    h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                             joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                             joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
    adj = adj_mx_from_skeleton(h36m_skeleton)
    model = SpatialTemporalModelOptimized1f(adj, num_joints_in=17, in_features=2, num_joints_out=17,
                                            filter_widths=[3, 3, 3], channels=128)
    model = model.cuda()

    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()

    print('INFO: Trainable parameter count:', model_params)
    input = torch.randn(2, 27, 17, 2)
    input = input.cuda()

    output = model(input)
    print(output.shape)
