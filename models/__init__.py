from .mobilevit import MobileViT


def create_mobilevit_model(cfg):
    exp_ratio = cfg['exp_ratio']
    last_layer_exp_factor = cfg['last_layer_exp_factor']
    num_heads = cfg['num_heads']
    depth = cfg['depth']
    mlp_ratio = cfg['mlp_ratio']
    embed_dim = cfg['embed_dim']
    channels = cfg['channels']
    return MobileViT(embed_dim=embed_dim, 
                     num_heads=num_heads,
                     depth=depth,
                     mlp_ratio=mlp_ratio,
                     channels=channels, 
                     exp_ratio=exp_ratio, 
                     last_layer_exp_factor=last_layer_exp_factor
                     )

# 1.27 M, 0.32 GMc
mobilevit_xxs_cfg = {
    'embed_dim': [64, 80, 96],
    'num_heads': 4,
    'depth': [2, 4, 3],
    'mlp_ratio': [2, 2, 2],
    'channels': [16, 24, 48, 64, 80],
    'exp_ratio': 2,
    'last_layer_exp_factor': 4,    
}

# 1.99 M, 0.65 GMc
mobilevit_mini_cfg = {
    'embed_dim': [80, 96, 128],
    'num_heads': 4,
    'depth': [2, 4, 3],
    'mlp_ratio': [2, 2, 2],
    'channels': [32, 48, 64, 80, 96],
    'exp_ratio': 4,
    'last_layer_exp_factor': 4,    
}

# 2.23 M, 0.67 GMc
mobilevit_xs_cfg = {
    'embed_dim': [96, 120, 144],
    'num_heads': 4,
    'depth': [2, 4, 3],
    'mlp_ratio': [2, 2, 2],
    'channels': [32, 48, 64, 80, 96],
    'exp_ratio': 4,
    'last_layer_exp_factor': 4,    
}

# 5.58 M, 1.76 GMc
mobilevit_s_cfg = {
    'embed_dim': [144, 192, 240],
    'num_heads': 4,
    'depth': [2, 4, 3],
    'mlp_ratio': [2, 2, 2],
    'channels': [32, 64, 96, 128, 160],
    'exp_ratio': 4,
    'last_layer_exp_factor': 4,    
}


model_cfg = {
    'mobilevit_xxs': mobilevit_xxs_cfg, 
    'mobilevit_mini': mobilevit_mini_cfg,
    'mobilevit_xs': mobilevit_xs_cfg, 
    'mobilevit_s': mobilevit_s_cfg, 
    }
