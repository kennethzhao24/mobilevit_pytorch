import math
import torch
from torch import nn


class MHSA(nn.Module):
    """
        Multi-head self attention: https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 attn_dropout = 0.0, 
                 bias = True
                 ):
        """
            :param embed_dim: embedding dimension
            :param num_heads: number of attention heads
            :param attn_dropout: attention dropout
            :param bias: use bias or not
        """
        super(MHSA, self).__init__()
        assert embed_dim % num_heads == 0, "Got: embed_dim={} and num_heads={}".format(embed_dim, num_heads)

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3*embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # [B x N x C]
        b_sz, n_patches, _ = x.shape

        # linear projection to qkv
        # [B x N x C] --> [B x N x 3 x h x C]
        qkv = (self.qkv_proj(x).reshape(b_sz, n_patches, 3, self.num_heads, -1))
        # [B x N x 3 x h x C] --> [B x h x 3 x N x C]
        qkv = qkv.transpose(1, 3)
        # [B x h x 3 x N x C] --> [B x h x N x C] x 3
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q * self.scaling
        # [B x h x N x C] --> [B x h x c x N]
        k = k.transpose(2, 3)

        # compute attention score
        # [B x h x N x c] x [B x h x c x N] --> [B x h x N x N]
        attn = torch.matmul(q, k)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [B x h x N x N] x [B x h x N x c] --> [B x h x N x c]
        out = torch.matmul(attn, v)
        # [B x h x N x c] --> [B x N x h x c] --> [B x N x C=ch]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out



class PositionalEncoding(nn.Module):
    """
        Sinusoidal positional embeddings (optional)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2) # [B x E x Max_patches)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(-1)]
        return self.dropout(x)
