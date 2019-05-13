Hybrid DenseNet-based CNN for Single Image Depth Estimation
============================================

This repository contains an updated version of the CNN from my repository [DE_resnet_unet_hyb](https://github.com/karoly-hars/DE_resnet_unet_hyb). The ResNet backbone has been replaced with a DenseNet169.

### Requirements
The code was tested with:
- python 3.5 and 3.6
- pytorch (and torchvision) 1.0.0
- opencv-python 3.4.3
- matplotlib 2.2.3
- numpy 1.15.4

### Guide
- Predicting the depth of an arbitrary image:
```sh
python3 predict_img.py <path_to_image>
```

### Evalutation
- Quantitative results on the NYU depth v2 test set:
 
| REL  |  RMSE  | Log10 |  δ1 |  δ2 |  δ3 |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0.129 | 0.588 | 0.056 |0.833 |0.962 |0.990 |


