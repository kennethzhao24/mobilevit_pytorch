# MobileViT 
PyTorch Implementation of [MobileViT](https://arxiv.org/pdf/2110.02178), with additional variation for downstream segmentation tasks.

### Supported Models
- [x] MobileViT 
- [x] MobileNetV2
- [x] ResNet

## Install
```bash
$ git clone https://github.com/kennethzhao24/mobilevit_pytorch.git
$ cd mobilevit_pytorch
$ pip install -r requirements.txt # install dependencies
$ pip install -e .
$ cd weights
$ sh download.sh # download pretrained weights
```
## Get-Started
```python
import argparse
import torch
from lib.models import build_model
parser = argparse.ArgumentParser(description='ImageNet Evaluation', add_help=True)
parser.add_argument('--config', type=str, default='./configs/imagenet.yaml', help="Configuration file")
parser.add_argument('--model_name', type=str, default='mobilevit_s', help="Model name")
opts = parser.parse_args()
opts = load_config_file(opts)

model = build_model(opts)

x = torch.randn(5, 3, 224, 224)
y = model(x)

print(y.shape)
```

## Evaluate model on ImageNet dataset
```bash
$ CUDA_VISIBLE_DEVICES=0 python main_eval.py \
   --model_name mobilevit_s \
   --weights ./weights/mobilevit_s.pt
```
### ImageNet-1k Results

|       Model    |  Parameters  |  Top-1 | Top-5  | 
| -------------  | ------------ | ------ | ------ | 
| MobileViT-XXS  |     1.3 M    |  69.0  |  88.9  | 
| MobileViT-Mini |     1.9 M    |  73.9  |  91.9  | 
|  MobileViT-XS  |     2.3 M    |  74.7  |  92.3  | 
|  MobileViT-S   |     5.6 M    |  78.3  |  94.1  |
|  MobileNetV2   |     3.5 M    |  73.5  |  91.6  | 
|    ResNet-50   |    25.6 M    |  78.6  |  94.5  | 



## TODO
- [ ] add other light-weight models
- [ ] add CIFAR 10/100 training
- [ ] add segmentation evaluation
