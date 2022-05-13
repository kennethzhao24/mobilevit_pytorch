from torch import nn
from ..layers import ConvBlock, Identity, get_activation_fn


class BasicResNetBlock(nn.Module):
    """
        ResNet Basic Block
    """
    expansion = 1

    def __init__(self, 
                 opts,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride = 1,
                 dilation = 1
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param mid_channels: number of middle layer channels
            :param out_channels: number of output channels
            :param stride: move the kernel by this amount during convolution operation
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.
        """
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)

        cbr_1 = ConvBlock(opts=opts, in_channels=in_channels, out_channels=mid_channels,
                          kernel_size=3, stride=stride, dilation=dilation, use_norm=True, use_act=True)
        cb_2 = ConvBlock(opts=opts, in_channels=mid_channels, out_channels=out_channels,
                         kernel_size=3, stride=1, use_norm=True, use_act=False, dilation=dilation)

        block = nn.Sequential()
        block.add_module(name="conv_batch_act_1", module=cbr_1)
        block.add_module(name="conv_batch_2", module=cb_2)

        down_sample = Identity()
        if stride == 2:
            down_sample = ConvBlock(opts=opts, in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, stride=stride, use_norm=True, use_act=False)

        super(BasicResNetBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.block = block
        self.down_sample = down_sample

        self.final_act = get_activation_fn(act_type=act_type, inplace=inplace,
                                           negative_slope=neg_slope, num_parameters=out_channels)

    def forward(self, x):
        out = self.block(x)
        res = self.down_sample(x)
        out = out + res
        return self.final_act(out)


class BottleneckResNetBlock(nn.Module):
    """
        ResNet Bottleneck Block
    """
    expansion = 4

    def __init__(self, 
                 opts,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride = 1,
                 dilation = 1
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param mid_channels: number of middle layer channels
            :param out_channels: number of output channels
            :param stride: move the kernel by this amount during convolution operation
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.
        """
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)

        cbr_1 = ConvBlock(opts=opts, in_channels=in_channels, out_channels=mid_channels,
                          kernel_size=1, stride=1, use_norm=True, use_act=True)
        cbr_2 = ConvBlock(opts=opts, in_channels=mid_channels, out_channels=mid_channels,
                          kernel_size=3, stride=stride, use_norm=True, use_act=True, dilation=dilation)
        cb_3 = ConvBlock(opts=opts, in_channels=mid_channels, out_channels=out_channels,
                         kernel_size=1, stride=1, use_norm=True, use_act=False)
        block = nn.Sequential()
        block.add_module(name="conv_batch_act_1", module=cbr_1)
        block.add_module(name="conv_batch_act_2", module=cbr_2)
        block.add_module(name="conv_batch_3", module=cb_3)

        down_sample = Identity()
        if stride == 2:
            down_sample = ConvBlock(opts=opts, in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, stride=stride, use_norm=True, use_act=False)
        elif in_channels != out_channels:
            down_sample = ConvBlock(opts=opts, in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, stride=1, use_norm=True, use_act=False)

        super(BottleneckResNetBlock, self).__init__()
        self.block = block

        self.down_sample = down_sample
        self.final_act = get_activation_fn(act_type=act_type,
                                           inplace=inplace,
                                           negative_slope=neg_slope,
                                           num_parameters=out_channels)


    def forward(self, x):
        out = self.block(x)
        res = self.down_sample(x)
        out = out + res
        return self.final_act(out)
