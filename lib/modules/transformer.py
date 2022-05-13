import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import ConvBlock, get_normalization_layer, get_activation_fn, MHSA



class TransformerEncoder(nn.Module):
    """
        Transfomer Encoder
    """
    def __init__(self, 
                 opts, 
                 embed_dim, 
                 ffn_latent_dim, 
                 num_heads = 8, 
                 attn_dropout = 0.0,
                 dropout = 0.1, 
                 ffn_dropout = 0.0,
                 transformer_norm_layer = "layer_norm",
                 ):
        """
            :param opts: arguments
            :param embed_dim: embedding dimension
            :param ffn_latent_dim: latent dimension of feedforward layer
            :param num_heads: Number of attention heads
            :param attn_dropout: attention dropout rate
            :param dropout: dropout rate
            :param ffn_dropout: feedforward dropout rate
            :param transformer_norm_layer: transformer norm layer
        """
        super(TransformerEncoder, self).__init__()

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            MHSA(embed_dim, num_heads, attn_dropout=attn_dropout, bias=True),
            nn.Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            self.build_act_layer(opts=opts),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )

    @staticmethod
    def build_act_layer(opts):
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(act_type=act_type, inplace=inplace, negative_slope=neg_slope,
                                      num_parameters=1)
        return act_layer

    def forward(self, x):
        # Multi-head attention
        x = x + self.pre_norm_mha(x)
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x




class MobileViTBlock(nn.Module):
    """
        MobileViT block: https://arxiv.org/pdf/2110.02178
    """
    def __init__(self, 
                 opts, 
                 in_channels, 
                 transformer_dim, 
                 ffn_dim,
                 n_transformer_blocks = 2,
                 head_dim = 32, 
                 attn_dropout = 0.1,
                 dropout = 0.1, 
                 ffn_dropout = 0.1, 
                 patch_h = 8,
                 patch_w = 8, 
                 transformer_norm_layer = "layer_norm",
                 conv_ksize = 3,
                 dilation = 1, 
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param transformer_dim: dimension of transformer encoder
            :param ffn_dim: dimension of feedforward layer
            :param n_transformer_block: number of transformer blocks
            :param head_dim: transformer head dimension     
            :param attn_dropout: Attention dropout     
            :param dropout: dropout
            :param ffn_dropout: feedforward dropout
            :param patch_h: split patch height size      
            :param patch_w: split patch width size
            :param transformer_norm_layer: transformer norm layer    
            :param conv_ksize: kernel size for convolutional block    
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.    

        """

        conv_3x3_in = ConvBlock(
            opts=opts, in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, dilation=dilation
        )
        conv_1x1_in = ConvBlock(
            opts=opts, in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, use_norm=False, use_act=False
        )

        conv_1x1_out = ConvBlock(
            opts=opts, in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1, use_norm=True, use_act=True
        )
        conv_3x3_out = ConvBlock(
            opts=opts, in_channels=2 * in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True
        )
        
        super(MobileViTBlock, self).__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [
            TransformerEncoder(opts=opts, embed_dim=transformer_dim, ffn_latent_dim=ffn_dims[block_idx], num_heads=num_heads,
                               attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout,
                               transformer_norm_layer=transformer_norm_layer)
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=transformer_dim)
        )
        self.global_rep = nn.Sequential(*global_rep)
        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w # n_w
        num_patch_h = new_h // patch_h # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, _, _, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x):
        res = x
        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)
        # learn global representations
        patches = self.global_rep(patches)
        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)
        fm = self.fusion(torch.cat((res, fm), dim=1))

        return fm