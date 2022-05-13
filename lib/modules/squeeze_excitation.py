from torch import nn
from ..layers import ConvBlock, get_activation_fn
from utils.math_utils import make_divisible


class SqueezeExcitation(nn.Module):
    """
        Squeeze excitation module
    """
    def __init__(self,
                 opts,
                 in_channels,
                 squeeze_factor = 4,
                 scale_fn_name = 'sigmoid'
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param squeeze_factor: squeeze factor
            :param scal_fn_name: activation layer
        """
        squeeze_channels = max(make_divisible(in_channels // squeeze_factor, 8), 32)

        fc1 = ConvBlock(opts=opts, in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1, stride=1,
                        bias=True, use_norm=False, use_act=True)
        fc2 = ConvBlock(opts=opts, in_channels=squeeze_channels, out_channels=in_channels, kernel_size=1, stride=1,
                        bias=True, use_norm=False, use_act=False)
        if scale_fn_name == "sigmoid":
            act_fn = get_activation_fn(act_type="sigmoid")
        elif scale_fn_name == "hard_sigmoid":
            act_fn = get_activation_fn(act_type="hard_sigmoid", inplace=True)
        else:
            raise NotImplementedError

        super(SqueezeExcitation, self).__init__()
        self.se_layer = nn.Sequential()
        self.se_layer.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(output_size=1))
        self.se_layer.add_module(name="fc1", module=fc1)
        self.se_layer.add_module(name="fc2", module=fc2)
        self.se_layer.add_module(name="scale_act", module=act_fn)

        self.in_channels = in_channels
        self.squeeze_factor = squeeze_factor
        self.scale_fn = scale_fn_name

    def forward(self, x):
        return x * self.se_layer(x)
