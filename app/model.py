# -*-coding:utf-8 -*-
# File       : model.py
# Time       : 2022/5/10 9:00
# Author     : pangyafei
# Description:
import torch
from torch import nn


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    m = M()
    input = torch.ones((64, 3, 32, 32))
    output = m(input)
    print(output.shape)

    print(torch.cuda.is_available())
