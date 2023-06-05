# -*-coding:utf-8 -*-
# File       : test_nn_model_load.py
# Time       : 2022/4/29 17:12
# Author     : pangyafei
# Description:

import torch
import torchvision.models

vgg16 = torch.load('vgg16.pth')
print(vgg16)

vgg16_2 = torchvision.models.vgg16(pretrained=False)
vgg16_2.load_state_dict("vgg16_2.pth")
print(vgg16_2)
