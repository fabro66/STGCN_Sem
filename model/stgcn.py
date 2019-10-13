import torch
import torch.nn as nn
from model.sem_gcn import SemGCN


class TemporalModelBase(nn.Module):
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
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)

        self.shrink = nn.Conv2d(channels, 3, 1, 1)

        self.hid_dim = 256

        self.semgcn = SemGCN(adj, channels, self.hid_dim, num_layers=4, non_local=True, p_dropout=None)

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
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)
        out_0 = self.shrink(x)
        out_0 = out_0.permute(0, 2, 3, 1)

        x = x.permute(0, 2, 3, 1)
        out_1, out_2 = self.semgcn(x)

        return out_0, out_1, out_2


class SpatialTemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, adj, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
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

        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []
        layers_res_conv = []
        layers_res_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            if i == len(filter_widths)-1:
                for j in range(self.num_joints_out):
                    layers_conv.append(nn.Conv1d(channels//4, channels//4,
                                                 filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                                                 dilation=next_dilation if not dense else 1,
                                                 bias=False))
                    layers_bn.append(nn.BatchNorm1d(channels//4, momentum=0.1))
                    layers_conv.append(nn.Conv1d(channels//4, channels, 1, dilation=1, bias=False))
                    layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

                    layers_res_conv.append(nn.Conv1d(channels, channels//4, 1, 1, bias=False))
                    layers_res_bn.append(nn.BatchNorm1d(channels//4))

                break

            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_res_conv = nn.ModuleList(layers_res_conv)
        self.layers_res_bn = nn.ModuleList(layers_res_bn)

    def _forward_blocks(self, x):
        t_size = x.shape[2]
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        b, c, _ = x.shape
        t_size = t_size - 3 ** len(self.filter_widths) + 1
        num_joints_features = torch.zeros(b, c, t_size, self.num_joints_out)
        num_joints_features = torch.cuda.comm.broadcast(num_joints_features.type(torch.cuda.FloatTensor), devices=[x.device.index])[0]

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]

            if i == len(self.pad)-2:
                for j in range(self.num_joints_out):
                    y = self.layers_res_bn[j](self.layers_res_conv[j](x))

                    y = self.drop(self.relu(self.layers_bn[2 * (i+j)](self.layers_conv[2 * (i+j)](y))))
                    y = res + self.drop(self.relu(self.layers_bn[2 * (i+j) + 1](self.layers_conv[2 * (i+j) + 1](y))))

                    y = y.unsqueeze(dim=-1)
                    num_joints_features[:, :, :, j:j+1] = y

                break

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        # x = self.shrink(x)
        return num_joints_features


class SpatialTemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, adj, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
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

        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], stride=filter_widths[0],
                                     bias=False)

        layers_conv = []
        layers_bn = []
        layers_res_conv = []
        layers_res_bn = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            if i == len(filter_widths)-1:
                for j in range(self.num_joints_out):
                    layers_conv.append(nn.Conv1d(channels//4, channels//4, filter_widths[i], stride=filter_widths[i], bias=False))
                    layers_bn.append(nn.BatchNorm1d(channels//4, momentum=0.1))
                    layers_conv.append(nn.Conv1d(channels//4, channels, 1, dilation=1, bias=False))
                    layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

                    layers_res_conv.append(nn.Conv1d(channels, channels // 4, 1, 1, bias=False))
                    layers_res_bn.append(nn.BatchNorm1d(channels // 4))
                break

            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_res_conv = nn.ModuleList(layers_res_conv)
        self.layers_res_bn = nn.ModuleList(layers_res_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        b, c, _ = x.shape
        num_joints_features = torch.zeros(b, c, 1, self.num_joints_out)
        num_joints_features = torch.cuda.comm.broadcast(num_joints_features.type(torch.cuda.FloatTensor), devices=[x.device.index])[0]

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2:: self.filter_widths[i + 1]]

            if i == len(self.pad)-2:
                for j in range(self.num_joints_out):
                    y = self.layers_res_bn[j](self.layers_res_conv[j](x))

                    y = self.drop(self.relu(self.layers_bn[2 * (i+j)](self.layers_conv[2 * (i+j)](y))))
                    y = res + self.drop(self.relu(self.layers_bn[2 * (i+j) + 1](self.layers_conv[2 * (i+j) + 1](y))))

                    y = y.unsqueeze(dim=-1)
                    num_joints_features[:, :, :, j:j+1] = y

                break

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        # x = self.shrink(x)
        return num_joints_features


if __name__ == "__main__":
    from common.skeleton import Skeleton
    from common.graph_utils import adj_mx_from_skeleton

    h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                             joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                             joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
    adj = adj_mx_from_skeleton(h36m_skeleton)
    model = SpatialTemporalModelOptimized1f(adj=adj, num_joints_in=17, in_features=2, num_joints_out=17, filter_widths=[3, 3, 3], channels=512)
    model = model.cuda()

    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    input = torch.randn(2, 28, 17, 2)
    input = input.cuda()

    x, y, z = model(input)
    print(x.shape, y.shape, z.shape)
