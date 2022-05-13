

mobilevit_xxs_cfg = {
    "layer1": {
        "out_channels": 16,
        "expand_ratio": 2,
        "num_blocks": 1,
        "stride": 1,
        "block_type": "mv2"
    },
    "layer2": {
        "out_channels": 24,
        "expand_ratio": 2,
        "num_blocks": 3,
        "stride": 2,
        "block_type": "mv2"
    },
    "layer3": {  # 28x28
        "out_channels": 48,
        "transformer_channels": 64,
        "ffn_dim": 128,
        "transformer_blocks": 2,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 2,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer4": {  # 14x14
        "out_channels": 64,
        "transformer_channels": 80,
        "ffn_dim": 160,
        "transformer_blocks": 4,
        "patch_h": 2, 
        "patch_w": 2, 
        "stride": 2,
        "mv_expand_ratio": 2,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer5": {  # 7x7
        "out_channels": 80,
        "transformer_channels": 96,
        "ffn_dim": 192,
        "transformer_blocks": 3,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 2,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "last_layer_exp_factor": 4
}

mobilevit_xs_cfg = {
    "layer1": {
        "out_channels": 32,
        "expand_ratio": 4,
        "num_blocks": 1,
        "stride": 1,
        "block_type": "mv2"
    },
    "layer2": {
        "out_channels": 48,
        "expand_ratio": 4,
        "num_blocks": 3,
        "stride": 2,
        "block_type": "mv2"
    },
    "layer3": {  # 28x28
        "out_channels": 64,
        "transformer_channels": 96,
        "ffn_dim": 192,
        "transformer_blocks": 2,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer4": {  # 14x14
        "out_channels": 80,
        "transformer_channels": 120,
        "ffn_dim": 240,
        "transformer_blocks": 4,
        "patch_h": 2, 
        "patch_w": 2, 
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer5": {  # 7x7
        "out_channels": 96,
        "transformer_channels": 144,
        "ffn_dim": 288,
        "transformer_blocks": 3,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "last_layer_exp_factor": 4
}

mobilevit_mini_cfg = {
    "layer1": {
        "out_channels": 32,
        "expand_ratio": 4,
        "num_blocks": 1,
        "stride": 1,
        "block_type": "mv2"
    },
    "layer2": {
        "out_channels": 48,
        "expand_ratio": 4,
        "num_blocks": 3,
        "stride": 2,
        "block_type": "mv2"
    },
    "layer3": {  # 28x28
        "out_channels": 64,
        "transformer_channels": 80,
        "ffn_dim": 160,
        "transformer_blocks": 2,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer4": {  # 14x14
        "out_channels": 80,
        "transformer_channels": 96,
        "ffn_dim": 192,
        "transformer_blocks": 4,
        "patch_h": 2, 
        "patch_w": 2, 
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer5": {  # 7x7
        "out_channels": 96,
        "transformer_channels": 128,
        "ffn_dim": 256,
        "transformer_blocks": 3,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "last_layer_exp_factor": 4
}

mobilevit_s_cfg = {
    "layer1": {
        "out_channels": 32,
        "expand_ratio": 4,
        "num_blocks": 1,
        "stride": 1,
        "block_type": "mv2"
    },
    "layer2": {
        "out_channels": 64,
        "expand_ratio": 4,
        "num_blocks": 3,
        "stride": 2,
        "block_type": "mv2"
    },
    "layer3": {  # 28x28
        "out_channels": 96,
        "transformer_channels": 144,
        "ffn_dim": 288,
        "transformer_blocks": 2,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer4": {  # 14x14
        "out_channels": 128,
        "transformer_channels": 192,
        "ffn_dim": 384,
        "transformer_blocks": 4,
        "patch_h": 2, 
        "patch_w": 2, 
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer5": {  # 7x7
        "out_channels": 160,
        "transformer_channels": 240,
        "ffn_dim": 480,
        "transformer_blocks": 3,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "last_layer_exp_factor": 4
}


resnet50_cfg = {
    'layer2': {
        'num_blocks':3,
        'mid_channels': 64,
        'block_type': 'bottleneck',
        'stride': 1
        },
    'layer3': {
        'num_blocks':4,
        'mid_channels': 128,
        'block_type': 'bottleneck',
        'stride': 2
        },
    'layer4': {
        'num_blocks':6,
        'mid_channels': 256,
        'block_type': 'bottleneck',
        'stride': 2
        },
    'layer5': {
        'num_blocks':3,
        'mid_channels': 512,
        'block_type': 'bottleneck',
        'stride': 2
        }       
}


mobilenetv2_config = {
        'layer1': {
            "expansion_ratio": 1,
            "out_channels": 16,
            "num_blocks": 1,
            "stride": 1
        },
        'layer2': {
            "expansion_ratio": 6,
            "out_channels": 24,
            "num_blocks": 2,
            "stride": 2
        },
        'layer3': {
            "expansion_ratio": 6,
            "out_channels": 32,
            "num_blocks": 3,
            "stride": 2
        },
        'layer4': {
            "expansion_ratio": 6,
            "out_channels": 64,
            "num_blocks": 4,
            "stride": 2
        },
        'layer4_a': {
            "expansion_ratio": 6,
            "out_channels": 96,
            "num_blocks": 3,
            "stride": 1
        },
        'layer5': {
            "expansion_ratio": 6,
            "out_channels": 160,
            "num_blocks": 3,
            "stride": 2
        },
        'layer5_a': {
            "expansion_ratio": 6,
            "out_channels": 320,
            "num_blocks": 1,
            "stride": 1
        }
}


model_cfg = {
    'resnet50': resnet50_cfg,
    'mobilenetv2': mobilenetv2_config,
    'mobilevit_xxs': mobilevit_xxs_cfg, 
    'mobilevit_mini': mobilevit_mini_cfg,
    'mobilevit_xs': mobilevit_xs_cfg, 
    'mobilevit_s': mobilevit_s_cfg, 
    }

    
def get_config(opts):
    model_name = getattr(opts, "model_name", 'mobilevit_s')
    return model_cfg[model_name]