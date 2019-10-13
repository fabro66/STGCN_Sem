import math
import torch
import torch.nn as nn


def _same_pad(k=1, dil=1):
    # assumes stride length of 1
    p = math.ceil(dil*(k - 1))
    return p


class TemResBlock(nn.Module):
    def __init__(self, channels, d, k=3, dropout=None, casual=False, use_bias=False,  final_layer=False):
        super(TemResBlock, self).__init__()
        self.channel = channels # input features
        self.d = d  # dilation size
        self.k = k  # "kernel size"
        self.drop = nn.Dropout(dropout)
        self.final_layer = final_layer
        ub = use_bias

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channels, channels//2, kernel_size=1, bias=ub)
        self.bn1 = nn.BatchNorm1d(channels//2, momentum=0.1)

        if casual:
            padding = (_same_pad(k, d), 0)
        else:
            p = _same_pad(k, d)
            if p % 2 == 1:
                padding = [p // 2 + 1, p // 2]
            else:
                padding = (p // 2, p // 2)
        self.pad = nn.ConstantPad1d(padding, 0.)

        if final_layer:
            self.final_pad = nn.ConstantPad1d((1, 1), 0.)

        self.dconv1 = nn.Conv1d(channels//2, channels//2, kernel_size=k, dilation=d, bias=ub)
        self.bn2 = nn.BatchNorm1d(channels // 2, momentum=0.1)

        self.conv2 = nn.Conv1d(channels//2, channels, kernel_size=1, bias=ub)
        self.bn3 = nn.BatchNorm1d(channels, momentum=0.1)

    def forward(self, input):
        x = input
        x = self.drop(self.relu(self.bn1(self.conv1(x))))

        if self.final_layer:
            x = self.final_pad(x)
            x[:, :, 0] = x[:, :, x.shape[-1]//2-7]
            x[:, :, -1] = x[:, :, x.shape[-1]//2+7]
            x = self.drop(self.relu(self.bn2(self.dconv1(x))))
            x = self.drop(self.relu(self.bn3(self.conv2(x))))

            padding = self.d // 2 - 1

            res = input[:, :, input.shape[-1]-padding::padding]
            x = x + res
            return x

        x = self.pad(x)
        x = self.drop(self.relu(self.bn2(self.dconv1(x))))
        x = self.drop(self.relu(self.bn3(self.conv2(x))))

        x = x + input  # add back in residual
        return x


class TemOpResBlock(nn.Module):
    def __init__(self, channels, d, k=3, dropout=None, casual=False, use_bias=False, final_layer=False):
        super(TemOpResBlock, self).__init__()
        self.channel = channels # input features
        self.d = d  # dilation size
        self.k = k  # "kernel size"
        self.drop = nn.Dropout(dropout)
        self.final_layer = final_layer
        ub = use_bias

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channels, channels//2, kernel_size=1, bias=ub)
        self.bn1 = nn.BatchNorm1d(channels//2, momentum=0.1)

        if casual:
            padding = (_same_pad(k, d), 0)
        else:
            p = _same_pad(k, d)
            if p % 2 == 1:
                padding = [p // 2 + 1, p // 2]
            else:
                padding = (p // 2, p // 2)
        self.pad = nn.ConstantPad1d(padding, 0.)

        if final_layer:
            self.dconv1 = nn.Conv1d(channels//2, channels//2, kernel_size=k, stride=1, bias=ub)
        else:
            self.dconv1 = nn.Conv1d(channels//2, channels//2, kernel_size=k, stride=2, bias=ub)
        self.op_pad_1 = nn.ConstantPad1d((1, 1), 0.)

        self.bn2 = nn.BatchNorm1d(channels // 2, momentum=0.1)

        self.conv2 = nn.Conv1d(channels//2, channels, kernel_size=1, bias=ub)
        self.bn3 = nn.BatchNorm1d(channels, momentum=0.1)

    def forward(self, input):
        x = input
        x = self.drop(self.relu(self.bn1(self.conv1(x))))

        if self.final_layer:
            x = self.drop(self.relu(self.bn2(self.dconv1(x))))
            x = self.drop(self.relu(self.bn3(self.conv2(x))))
            res = input[:, :, 1::2]
            x = x + res

            return x

        x = self.op_pad_1(x)
        res = self.op_pad_1(input)

        x = self.drop(self.relu(self.bn2(self.dconv1(x))))
        x = self.drop(self.relu(self.bn3(self.conv2(x))))
        res_ = res[:, :, 1::2]
        x = x + res_
        print(x.shape)
        return x


class ResBlockSet(nn.Module):
    def __init__(self, channels, filter_widths, kernel=3, casual=False,
                 dropout=None, num_sets=1, final_set=False, optimizer=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.final_set = final_set
        self.optimizer = optimizer
        self.num_sets = num_sets

        dil_num = 2 ** (len(filter_widths) + 1)
        dlist = [1 << x for x in range(dil_num - 1) if (1 << x) <= dil_num]

        if final_set:
            if num_sets == 1:
                dlist = dlist[1:-1]
            else:
                dlist = dlist[:-1]

            if optimizer and num_sets > 1:
                self.expand_conv1 = nn.Conv1d(channels, channels//2, 1, bias=False)
                self.expand_bn1 = nn.BatchNorm1d(channels//2)
                self.expand_dconv = nn.Conv1d(channels//2, channels//2, kernel, stride=2, padding=2, bias=False)
                self.expand_bn2 = nn.BatchNorm1d(channels//2, momentum=0.1)
                self.expand_conv2 = nn.Conv1d(channels//2, channels, 1, bias=False)
                self.expand_bn3 = nn.BatchNorm1d(channels)
                self.relu = nn.ReLU(inplace=True)
                dlist = dlist[1:]

            print(dlist)
            if optimizer:
                self.blocks = nn.Sequential(*[TemOpResBlock(channels, d, kernel, casual) for d in dlist[:-1]])
                self.final_layer = TemOpResBlock(channels, dlist[-1], kernel, casual, final_layer=True)

            else:
                self.blocks = nn.Sequential(*[TemResBlock(channels, d, kernel, casual) for d in dlist[:-1]])
                self.final_layer = TemResBlock(channels, dlist[-1], kernel, casual, final_layer=True)
        else:
            print(dlist)
            dlist = dlist[1:]
            self.blocks = nn.Sequential(*[TemResBlock(channels, d, kernel, casual) for d in dlist])

    def forward(self, x):
        if self.final_set and self.optimizer and self.num_sets > 1:
            x = self.drop(self.relu(self.expand_bn1(self.expand_conv1(x))))
            x = self.drop(self.relu(self.expand_bn2(self.expand_dconv(x))))
            x = self.drop(self.relu(self.expand_bn3(self.expand_conv2(x))))

        x = self.blocks(x)

        if self.final_set:
            x = self.drop(self.final_layer(x))
        return x


if __name__ == "__main__":
    # from common.skeleton import Skeleton
    # from common.graph_utils import adj_mx_from_skeleton
    import torch

    # h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
    #                          joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    #                          joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
    # adj = adj_mx_from_skeleton(h36m_skeleton)
    model = ResBlockSet(channels=1024, filter_widths=[2, 2, 2], kernel=3, num_sets=2, final_set=True, optimizer=True)
    model = model.cuda()

    input = torch.randn(2, 1024, 15)
    input = input.cuda()

    model_params = 0
    output = model(input)
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print(output.shape)
