# MobileViT
PyTorch Implementation of [MobileViT](https://arxiv.org/pdf/2110.02178), with additional variation for downstream segmentation tasks.

- [x] MobileViT
- [ ] ImageNet Evaluation
- [ ] CIFAR 10/100 Training
- [ ] Segmentation Evaluation (mmseg)


## ImageNet-1k classification


|       Model     |    Params    |  Top-1 | Top-5  |
| --------------  | ------------ | ------ | ------ | 
|  MobileViT-XXS  |     1.3 M    |  69.0  |  88.9  |  
|  MobileViT-XS   |     2.3 M    |  74.7  |  92.3  |
|  MobileViT-S    |     5.6 M    |  78.3  |  94.1  | 
| MobileViT-Mini* |     2.0 M    |  73.9  |  91.9  | 


## Semantic segmentation (mmseg)

|   Backbone     |   Decoder Head   |    Dataset   |  Params  |  mIoU | 
| -------------  | -----------------| ------------ | -------- | ----- | 
|  MobileViT-S   |  SegFormerHead   |    ADE20K    |   5.3 M  | 33.23 | 
|  MobileViT-S   |  SegFormerHead   |  CityScapes  |   5.3 M  | 73.4  | 
