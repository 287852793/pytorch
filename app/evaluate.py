# -*-coding:utf-8 -*-
# File       : evaluate.py
# Time       : 2022/5/12 10:50
# Author     : pangyafei
# Description:

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_data_size = len(test_data)

test_dataloader = DataLoader(test_data, batch_size=64)

loss_fn = nn.CrossEntropyLoss()

m = torch.load("./models/m_49.pth", map_location=torch.device('cpu'))

total_test_loss = 0
total_accuracy = 0
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        outputs = m(imgs)
        loss = loss_fn(outputs, targets)
        total_test_loss = total_test_loss + loss
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy = total_accuracy + accuracy

    print("整体测试集上的 Loss： {}".format(total_test_loss))
    print("整体测试集上的正确率： {}".format(total_accuracy / test_data_size))
