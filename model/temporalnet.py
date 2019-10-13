import torch
import math
import torch.nn as nn
from model.resblock import ResBlockSet


def _same_pad(k=1, dil=1):
    # assumes stride length of 1
    p = math.ceil(dil*(k - 1))
    return p


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths,
                 causal, dropout, channels):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 == 0, 'Only Even filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        if dropout is not None:
            self.drop = nn.Dropout(dropout)

        self.relu = nn.ReLU(inplace=True)
        self.expand_bn = nn.BatchNorm1d(channels)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    # def set_bn_momentum(self, momentum):
    #     self.expand_bn.momentum = momentum
    #     for bn in self.layers_bn:
    #         bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 2**(len(self.filter_widths) + 1) - 1
        return frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = 2 ** (len(self.filter_widths) + 1) // 2
        return frames

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)
        x = self.shrink(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)

        return x


class ByteModel(TemporalModelBase):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths,
                 causal=False, dropout=0.25, channels=1024, kernel=3, num_sets=1):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        assert num_sets <= 2, "Sure num_sets <= 2!"

        self.expand_conv = nn.Conv1d(in_features*num_joints_in, channels, 3, padding=1, bias=False)

        self.sets = nn.Sequential()
        if num_sets > 1:
            for i in range(num_sets-1):
                self.sets.add_module("set_{}".format(i+1), ResBlockSet(channels, filter_widths, kernel,
                                                                       causal, dropout, num_sets))

        self.sets.add_module("set_{}".format(num_sets), ResBlockSet(channels, filter_widths, kernel,
                                                                    causal, dropout, num_sets, final_set=True))

    def _forward_blocks(self, x):
        x = self.expand_conv(x)
        x = self.sets(x)
        return x


class ByteModelOptimized1f(TemporalModelBase):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths,
                 causal=False, dropout=0.25, channels=1024, kernel=3, num_sets=1):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        assert num_sets <= 2, "Sure num_sets <= 2!"

        self.num_sets = num_sets
        self.sets = nn.Sequential()
        if num_sets > 1:
            self.expand_conv = nn.Conv1d(in_features * num_joints_in, channels, kernel, padding=1, bias=False)
            for i in range(num_sets-1):
                self.sets.add_module("set_{}".format(i+1), ResBlockSet(channels, filter_widths, kernel,
                                                                       causal, dropout, num_sets))
        else:
            self.expand_conv = nn.Conv1d(in_features * num_joints_in, channels, kernel, stride=2, padding=2, bias=False)

        self.sets.add_module("set_{}".format(num_sets), ResBlockSet(channels, filter_widths, kernel, causal, dropout,
                                                                    num_sets, final_set=True, optimizer=True))

    def _forward_blocks(self, x):
        x = self.expand_bn(self.expand_conv(x))
        x = self.sets(x)
        return x


if __name__ == "__main__":
    # from common.skeleton import Skeleton
    # from common.graph_utils import adj_mx_from_skeleton
    import torch
    from common.loss import mpjpe
    # h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
    #                          joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    #                          joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
    # adj = adj_mx_from_skeleton(h36m_skeleton)
    model = ByteModel(num_joints_in=17, in_features=2, num_joints_out=17, filter_widths=[2, 2, 2],
                                 channels=1024, num_sets=1)
    model = model.cuda()

    input = torch.randn(2, 15, 17, 2)
    input = input.cuda()
    target_3d = torch.randn(2, 1, 17, 3)
    target_3d = target_3d.cuda()

    model_params = 0
    output = model(input)
    # loss = mpjpe(output, target_3d)
    # loss.backward()
    for parameter in model.parameters():
        model_params += parameter.numel()

    print('INFO: Trainable parameter count:', model_params)
    print(output.shape)
