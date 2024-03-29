Hybrid DenseNet-based CNN for Single Image Depth Estimation
============================================

This repository contains an updated version of the CNN from my repository [DE_resnet_unet_hyb](https://github.com/karoly-hars/DE_resnet_unet_hyb). The ResNet backbone has been replaced with a DenseNet169.

### Requirements
The code was tested with:
- python 3.5 and 3.6
- pytorch (and torchvision) 1.3.0
- opencv-python 3.4.3
- matplotlib 2.2.3
- numpy 1.15.4

### Guide
- Predicting the depth of an arbitrary image:
```
python3 predict_img.py -i <path_to_image> -o <path_to_output>
```

### Evalutation
- Quantitative results on the NYU depth v2 test set:
 
| REL  |  RMSE  | Log10 |  δ1 |  δ2 |  δ3 |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0.129 | 0.588 | 0.056 |0.833 |0.962 |0.990 |


### About the training code
Lots of people reached out to me about the training code. Sadly, I stopped working on this project a long time ago.
I don't have access to the same data and codebase anymore, so can't share the training code. However, the work is based on this paper:
https://link.springer.com/chapter/10.1007%2F978-3-319-98678-4_38, which describes the training process in detail,
and the depth dataset is available for researchers and students. The only difference compared to the article is the network structure, but it can be
copied from the `network.py` module. If anyone is willing to invest time into writing the training code for themselves, I am happy to help.
