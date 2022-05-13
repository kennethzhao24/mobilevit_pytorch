from torch import nn

from .utils import get_config
from ..layers import ConvBlock, GlobalPool
from ..modules import InvertedResidual, MobileViTBlock
from utils.model_utils import initialize_weights


class MobileViT(nn.Module):
    """
        MobileViT: https://arxiv.org/pdf/2110.02178
    """
    def __init__(self, opts):
        image_channels, input_channels = 3, 16
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.1)

        # original mobilevit uses swish activation function
        setattr(opts, "model.activation.name", "swish")

        mobilevit_config = get_config(opts=opts)

        super(MobileViT, self).__init__()

        self.dilation = 1
        self.conv_1 = ConvBlock(
                opts=opts, in_channels=image_channels, out_channels=input_channels,
                kernel_size=3, stride=2, use_norm=True, use_act=True
            )

        self.layer_1, self.layer_1_channels = self._make_layer(
            opts=opts, input_channel=input_channels, cfg=mobilevit_config["layer1"]
        )

        self.layer_2, self.layer_2_channels = self._make_layer(
            opts=opts, input_channel=self.layer_1_channels, cfg=mobilevit_config["layer2"]
        )

        self.layer_3, self.layer_3_channels = self._make_layer(
            opts=opts, input_channel=self.layer_2_channels, cfg=mobilevit_config["layer3"]
        )

        self.layer_4, self.layer_4_channels = self._make_layer(
            opts=opts, input_channel=self.layer_3_channels, cfg=mobilevit_config["layer4"],
        )

        self.layer_5, self.layer_5_channels = self._make_layer(
            opts=opts, input_channel=self.layer_4_channels, cfg=mobilevit_config["layer5"], 
        )

        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * self.layer_5_channels, 960)
        self.conv_1x1_exp = ConvBlock(
                opts=opts, in_channels=self.layer_5_channels, out_channels=exp_channels,
                kernel_size=1, stride=1, use_act=True, use_norm=True
            )

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool())
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=classifier_dropout, inplace=True))
        self.classifier.add_module(
            name="fc",
            module=nn.Linear(in_features=exp_channels, out_features=num_classes, bias=True)
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

    def _make_layer(self, opts, input_channel, cfg, dilate = False):
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg,
                dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel, cfg):
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            input_channel = output_channels
            block.append(layer)
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg, dilate = False):
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        block.append(
            MobileViTBlock(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.classification.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.classification.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(opts, "model.classification.mit.attn_dropout", 0.0),
                head_dim=head_dim,
                conv_ksize=getattr(opts, "model.classification.mit.conv_kernel_size", 3)
            )
        )

        return nn.Sequential(*block), input_channel
