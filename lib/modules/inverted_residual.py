import torch.nn as nn

from ..layers import ConvBlock, get_activation_fn
from .squeeze_excitation import SqueezeExcitation
from utils.math_utils import make_divisible


class InvertedResidualSE(nn.Module):
    """
        Inverted residual block w/ SE (MobileNetv3)
    """
    def __init__(self,
                 opts,
                 in_channels,
                 out_channels,
                 expand_ratio,
                 use_hs = False,
                 dilation = 1,
                 stride = 1,
                 use_se = False
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param expand_ratio: expand ratio for hidden dimension
            :param use_hs: use hard swish actiavtion or not
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.
            :param stride: move the kernel by this amount during convolution operation
            :param use_se: use squeeze excitation module or not
        """
        super(InvertedResidualSE, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if use_hs:
            act_fn = get_activation_fn(act_type="hard_swish", inplace= True)
        else:
            act_fn = get_activation_fn(act_type="relu", inplace=True)

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1",
                             module=ConvBlock(opts, in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                                              use_act=False, use_norm=True))
            block.add_module(name="act_fn_1", module=act_fn)

        block.add_module(
            name="conv_3x3",
            module=ConvBlock(opts, in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3,
                             groups=hidden_dim, use_act=False, use_norm=True, dilation=dilation)
        )
        block.add_module(name="act_fn_2", module=act_fn)

        if use_se:
            se = SqueezeExcitation(opts=opts, in_channels=hidden_dim, squeeze_factor=4, scale_fn_name="hard_sigmoid")
            block.add_module(name="se", module=se)

        block.add_module(name="red_1x1",
                         module=ConvBlock(opts, in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                                          use_act=False, use_norm=True))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.use_hs = use_hs
        self.use_se = use_se

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res_connect else y



class InvertedResidual(nn.Module):
    """
        Inverted residual block (MobileNetv2)
    """
    def __init__(self,
                 opts,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation = 1
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: move the kernel by this amount during convolution operation
            :param expand_ratio: expand ratio for hidden dimension
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.
        """
        assert stride in [1, 2]
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1",
                             module=ConvBlock(opts, in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                                              use_act=True, use_norm=True))

        block.add_module(
            name="conv_3x3",
            module=ConvBlock(opts, in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3,
                             groups=hidden_dim, use_act=True, use_norm=True, dilation=dilation)
        )

        block.add_module(name="red_1x1",
                         module=ConvBlock(opts, in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                                          use_act=False, use_norm=True))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)
