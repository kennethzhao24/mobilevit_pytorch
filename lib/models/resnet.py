from torch import nn

from .utils import get_config
from ..layers import ConvBlock, GlobalPool, Identity
from ..modules import BasicResNetBlock, BottleneckResNetBlock
from utils.model_utils import initialize_weights


class ResNet(nn.Module):
    """
        ResNet: https://arxiv.org/pdf/1512.03385.pdf
        Modifications to the original ResNet architecture
        1. First 7x7 strided conv is replaced with 3x3 strided conv
        2. MaxPool operation is replaced with another 3x3 strided depth-wise conv
    """
    def __init__(self, opts):
        image_channels, input_channels = 3, 64
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.1)

        # resnet uses swish relu function
        setattr(opts, "model.activation.name", "relu")

        resnet_config = get_config(opts=opts)

        super(ResNet, self).__init__()

        self.dilation = 1

        self.conv_1 = ConvBlock(opts=opts, in_channels=image_channels, out_channels=input_channels,
                                kernel_size=3, stride=2, use_norm=True, use_act=True)

        self.layer_1 = ConvBlock(opts=opts, in_channels=input_channels, out_channels=input_channels,
                                 kernel_size=3, stride=2, use_norm=True, use_act=True, groups=input_channels)

        self.layer_2, self.layer_2_channels = self._make_layer(opts=opts,
                                                      in_channels=input_channels,
                                                      layer_config=resnet_config["layer2"]
                                                      )

        self.layer_3, self.layer_3_channels = self._make_layer(opts=opts,
                                                      in_channels=self.layer_2_channels,
                                                      layer_config=resnet_config["layer3"]
                                                      )

        self.layer_4, self.layer_4_channels = self._make_layer(opts=opts,
                                                      in_channels=self.layer_3_channels,
                                                      layer_config=resnet_config["layer4"],
                                                      )

        self.layer_5, self.layer_5_channels = self._make_layer(opts=opts,
                                                      in_channels=self.layer_4_channels,
                                                      layer_config=resnet_config["layer5"],
                                                      )

        self.conv_1x1_exp = Identity()
        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool())
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="classifier_dropout", module=nn.Dropout(p=classifier_dropout))
        self.classifier.add_module(name="classifier_fc",
                                   module=nn.Linear(in_features=self.layer_5_channels, out_features=num_classes, bias=True)
                                   )

        # weight initialization
        self.reset_parameters(opts=opts)

    def reset_parameters(self, opts):
        initialize_weights(opts=opts, modules=self.modules())

    def extract_features(self, x):
        out_dict = {} # consider input size of 224
        x = self.conv_1(x) # 112 x112
        x = self.layer_1(x) # 112 x112
        out_dict["out_l1"] = x  # level-1 feature

        x = self.layer_2(x) # 56 x 56
        out_dict["out_l2"] = x

        x = self.layer_3(x) # 28 x 28
        out_dict["out_l3"] = x

        x = self.layer_4(x) # 14 x 14
        out_dict["out_l4"] = x

        x = self.layer_5(x) # 7 x 7
        out_dict["out_l5"] = x

        if self.conv_1x1_exp is not None:
            x = self.conv_1x1_exp(x) # 7 x 7
            out_dict["out_l5_exp"] = x

        return out_dict, x

    def forward(self, x):
        _, x = self.extract_features(x)
        x = self.classifier(x)
        return x

    def _make_layer(self, opts, in_channels, layer_config, dilate = False):

        if layer_config.get("block_type", "bottleneck").lower() == "bottleneck":
            block_type = BottleneckResNetBlock
        else:
            block_type = BasicResNetBlock

        mid_channels = layer_config.get("mid_channels")
        num_blocks = layer_config.get("num_blocks", 2)
        stride = layer_config.get("stride", 1)

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        out_channels = block_type.expansion * mid_channels

        block = nn.Sequential()
        block.add_module(
            name="block_0",
            module=block_type(opts=opts, in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels,
                              stride=stride, dilation=previous_dilation)
        )

        for block_idx in range(1, num_blocks):
            block.add_module(
                name="block_{}".format(block_idx),
                module=block_type(opts=opts, in_channels=out_channels, mid_channels=mid_channels,
                                  out_channels=out_channels,
                                  stride=1, dilation=self.dilation)
            )

        return block, out_channels

