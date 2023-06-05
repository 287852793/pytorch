# -*-coding:utf-8 -*-
# File       : test_nn_model_save.py
# Time       : 2022/4/29 17:10
# Author     : pangyafei
# Description:

import torch
import torchvision


vgg16 = torchvision.models.vgg16(pretrained=False)

# method 1 模型结构+模型参数
torch.save(vgg16, 'vgg16.pth')

# method 2 模型参数（官方推荐）
torch.save(vgg16.state_dict(), 'vgg16_2.pth')
