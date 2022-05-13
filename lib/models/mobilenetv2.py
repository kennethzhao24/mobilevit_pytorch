from typing import Dict
from torch import nn

from .utils import get_config
from ..layers import ConvBlock, GlobalPool
from ..modules import InvertedResidual
from utils.model_utils import initialize_weights
from utils.math_utils import make_divisible


class MobileNetV2(nn.Module):
    """
        MobileNetV2: https://arxiv.org/pdf/1801.04381.pdf
    """
    def __init__(self, opts):
        image_channels, input_channels, last_channel = 3, 32, 1280
        width_mult = getattr(opts, "model.classification.mobilenetv2.width-multiplier", 1.0)
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.2)

        # mobilenetv2 uses relu6 function
        setattr(opts, "model.activation.name", "relu6")

        mobilenetv2_config = get_config(opts=opts)

        super(MobileNetV2, self).__init__()

        self.dilation = 1

        self.conv_1 = ConvBlock(opts=opts, in_channels=image_channels, out_channels=input_channels,
                                kernel_size=3, stride=2, use_norm=True, use_act=True)

        self.layer_1, self.layer_1_channels = self._make_layer(opts=opts,
                                                      mv2_config=mobilenetv2_config['layer1'],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels)

        self.layer_2, self.layer_2_channels = self._make_layer(opts=opts,
                                                      mv2_config=mobilenetv2_config['layer2'],
                                                      width_mult=width_mult,
                                                      input_channel=self.layer_1_channels)

        self.layer_3, self.layer_3_channels = self._make_layer(opts=opts,
                                                      mv2_config=mobilenetv2_config['layer3'],
                                                      width_mult=width_mult,
                                                      input_channel=self.layer_2_channels)

        self.layer_4, self.layer_4_channels = self._make_layer(opts=opts,
                                                      mv2_config=[mobilenetv2_config['layer4'], mobilenetv2_config['layer4_a']],
                                                      width_mult=width_mult,
                                                      input_channel=self.layer_3_channels)

        self.layer_5, self.layer_5_channels = self._make_layer(opts=opts,
                                                      mv2_config=[mobilenetv2_config['layer5'], mobilenetv2_config['layer5_a']],
                                                      width_mult=width_mult,
                                                      input_channel=self.layer_4_channels)                                                    
        self.conv_1x1_exp =  ConvBlock(
                opts=opts, in_channels=self.layer_5_channels, out_channels=last_channel,
                kernel_size=1, stride=1, use_act=True, use_norm=True)

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool())
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="classifier_dropout", module=nn.Dropout(p=classifier_dropout))
        self.classifier.add_module(name="classifier_fc",
                                   module=nn.Linear(in_features=last_channel, out_features=num_classes, bias=True)
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

    def _make_layer(self, opts, mv2_config, width_mult, input_channel,
                    round_nearest = 8, dilate = False):
        prev_dilation = self.dilation
        mv2_block = nn.Sequential()
        count = 0

        if isinstance(mv2_config, Dict):
            mv2_config = [mv2_config]

        for cfg in mv2_config:
            t = cfg.get("expansion_ratio")
            c = cfg.get("out_channels")
            n = cfg.get("num_blocks")
            s = cfg.get("stride")

            output_channel = make_divisible(c * width_mult, round_nearest)

            for block_idx in range(n):
                stride = s if block_idx == 0 else 1
                block_name = "mv2_block_{}".format(count)
                if dilate and count == 0:
                    self.dilation *= stride
                    stride = 1

                layer = InvertedResidual(
                    opts=opts,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=stride,
                    expand_ratio=t,
                    dilation=prev_dilation if count == 0 else self.dilation
                )
                mv2_block.add_module(name=block_name, module=layer)
                count += 1
                input_channel = output_channel
        return mv2_block, input_channel