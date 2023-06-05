# -*-coding:utf-8 -*-
# File       : test_nn_model.py
# Time       : 2022/4/29 14:41
# Author     : pangyafei
# Description:

import torchvision
from torchvision.transforms import ToTensor

# trainset = torchvision.datasets.ImageNet("datasets", "train", transform=ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)