import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def conv_1x1_bn(in_channels, out_channels, norm=True):
    """
        1x1 Convolution Block
    """
    if norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
            )
    else:
        return nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)


def conv_3x3_bn(in_channels, out_channels, stride=1):
    """
        3x3 Convolution Block
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU()
    )


class FFN(nn.Module):
    """
        Feedforward (MLP) Block
    """
    def __init__(self, dim, hidden_dim, ffn_dropout=0.):
        super(FFN, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.block(x)


class MHSA(nn.Module):
    """
        Multi-Head Self-Attention: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, embed_dim, num_heads, attn_dropout=0.):
        super(MHSA, self).__init__()
        assert embed_dim % num_heads == 0
        self.qkv_proj = nn.Linear(embed_dim, embed_dim*3, bias=True)

        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.softmax = nn.Softmax(dim = -1)
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.out_proj(out)


class TransformerEncoder(nn.Module):
    """
        Transformer Enocder
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0., attn_dropout=0., ffn_dropout=0.):
        super(TransformerEncoder, self).__init__()
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            MHSA(embed_dim, num_heads, attn_dropout),
            nn.Dropout(dropout)
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            FFN(embed_dim, mlp_dim, ffn_dropout),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.pre_norm_mha(x)
        x = x + self.pre_norm_ffn(x)
        return x


class InvertedResidual(nn.Module):
    """
        Inverted Residual Block (MobileNetv2)
    """
    def __init__(self, in_channels, out_channels, stride=1, exp_ratio=4):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        
        self.stride = stride
        hidden_dim = int(in_channels * exp_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if exp_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    """
        MobileViT Block
    """
    def __init__(self, channels, embed_dim, depth, num_heads, mlp_dim, patch_size=(2,2), dropout=0.1):
        super(MobileViTBlock, self).__init__()
        self.ph, self.pw = patch_size
        self.conv_3x3_in = conv_3x3_bn(channels, channels)
        self.conv_1x1_in = conv_1x1_bn(channels, embed_dim, norm=False)
        transformer = nn.ModuleList([])
        for i in range(depth):
            transformer.append(TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout))
        transformer.append(nn.LayerNorm(embed_dim))
        self.transformer = nn.Sequential(*transformer)

        self.conv_1x1_out = conv_1x1_bn(embed_dim, channels, norm=True)
        self.conv_3x3_out = conv_3x3_bn(2 * channels, channels)
    
    def forward(self, x):
        _, _, h, w = x.shape
        # make sure to height and width are divisible by patch size
        new_h = int(math.ceil(h / self.ph) * self.ph)
        new_w = int(math.ceil(w / self.pw) * self.pw)
        if new_h != h and new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        y = x.clone()
        # Local representations
        x = self.conv_3x3_in(x)
        x = self.conv_1x1_in(x)
        # Global representations
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=new_h//self.ph, w=new_w//self.pw, ph=self.ph, pw=self.pw)
        # Fusion
        x = self.conv_1x1_out(x)
        x = torch.cat((x, y), 1)
        x = self.conv_3x3_out(x)
        return x


class MobileViT(nn.Module):
    """
        MobileViT: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(self, 
                 image_size=(224,224), 
                 embed_dim=[64, 80, 96],
                 num_heads=4,
                 depth=[2,4,3],
                 mlp_ratio=[2,2,2],
                 channels=[16,24,48,64,80], 
                 exp_ratio=4, 
                 last_layer_exp_factor=4,
                 patch_size=(2, 2),
                 classifier_dropout=0.1,
                 num_classes=1000,
                 ):
        super(MobileViT, self).__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        self.conv_in = conv_3x3_bn(3, 16, stride=2) # 112 x 112

        self.layer1 = InvertedResidual(16, channels[0], 1, exp_ratio)

        layer2 = nn.ModuleList([])
        layer2.append(InvertedResidual(channels[0], channels[1], 2, exp_ratio)) # 56 x 56
        layer2.append(InvertedResidual(channels[1], channels[1], 1, exp_ratio))
        layer2.append(InvertedResidual(channels[1], channels[1], 1, exp_ratio))
        self.layer2 = nn.Sequential(*layer2)

        layer3 = nn.ModuleList([])
        layer3.append(InvertedResidual(channels[1], channels[2], 2, exp_ratio)) # 28 x 28
        layer3.append(MobileViTBlock(channels[2],
                                     embed_dim[0], 
                                     depth[0], 
                                     num_heads,
                                     int(embed_dim[0] * mlp_ratio[0])
                                     ))
        self.layer3 = nn.Sequential(*layer3)

        layer4 = nn.ModuleList([])
        layer4.append(InvertedResidual(channels[2], channels[3], 2, exp_ratio)) # 14 x 14
        layer4.append(MobileViTBlock(channels[3],
                                     embed_dim[1], 
                                     depth[1], 
                                     num_heads,
                                     int(embed_dim[1] * mlp_ratio[1])
                                     ))
        self.layer4 = nn.Sequential(*layer4)

        layer5 = nn.ModuleList([])
        layer5.append(InvertedResidual(channels[3], channels[4], 2, exp_ratio)) # 7 x 7
        layer5.append(MobileViTBlock(channels[4],
                                     embed_dim[2], 
                                     depth[2], 
                                     num_heads,
                                     int(embed_dim[2] * mlp_ratio[2])
                                     ))
        self.layer5 = nn.Sequential(*layer5)

        self.conv_out = conv_1x1_bn(channels[4], channels[4] * last_layer_exp_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=classifier_dropout, inplace=True)
        self.classifier = nn.Linear(channels[4] * last_layer_exp_factor, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.conv_out(x)
        x = self.pool(x).view(-1, x.shape[1])
        x = self.classifier(x)
        return x

