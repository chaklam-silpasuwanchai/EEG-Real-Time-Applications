import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TypeVar, Union, Tuple

T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]


class ResidualBlock(nn.Module):
    """All credits to : trailingend
    Link to his/her repository : https://github.com/trailingend"""

    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()

        # TODO: 3x3 convolution -> relu
        # the input and output channel number is channel_num
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, stride=2),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_num),
        )
        self.relu = nn.ReLU()

        # self.test_conv = nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, stride=(2, 2))

    def forward(self, x):
        # TODO: forward
        residual = self.conv_block1(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        # out = self.test_conv(out)
        return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class SeparableConv2d(torch.nn.Module):
    """ All credits goes to 'bdsaglam' on github : https://gist.github.com/bdsaglam"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
                 ):
        super().__init__()

        intermediate_channels = in_channels * depth_multiplier
        self.depthwiseConv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )
        self.pointConv = torch.nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        return self.pointConv(self.depthwiseConv(x))


class ConstraintConv2d(torch.nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 weight_max_lim: float = None,
                 weight_min_lim: float = None,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',  # TODO: refine this type
                 device=None,
                 dtype=None):
        super(ConstraintConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride, padding=padding, dilation=dilation, groups=groups,
                                               bias=bias,
                                               padding_mode=padding_mode, device=device, dtype=dtype)
        self.w_max = weight_max_lim
        self.w_min = weight_min_lim

    def forward(self, x):
        return F.conv2d(x, self.weight.clamp(max=self.w_max, min=self.w_min), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConstraintLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, weight_max_lim: float = None, weight_min_lim: float = None,
                 bias: bool = True, device=None, dtype=None):
        super(ConstraintLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias,
                                               device=device, dtype=dtype)
        self.w_max = weight_max_lim
        self.w_min = weight_min_lim

    def forward(self, x):
        return F.linear(x, self.weight.clamp(max=self.w_max, min=self.w_min), self.bias)
