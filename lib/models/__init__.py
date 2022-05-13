from .mobilenetv2 import MobileNetV2
from .resnet import ResNet
from .vit import MobileViT


def build_model(opts):
    model_name = getattr(opts, "model_name")
    if 'resnet' in model_name:
        model = ResNet(opts=opts)
    elif 'mobilevit' in model_name:
        model = MobileViT(opts=opts)
    elif 'mobilenet' in model_name:
        model = MobileNetV2(opts=opts)
    else:
        raise ValueError('Model Not Supported!')
    return model
