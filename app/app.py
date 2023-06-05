# -*-coding:utf-8 -*-
# File       : app.py
# Time       : 2022/5/10 8:46
# Author     : pangyafei
# Description:
import time
from builtins import print

import torch
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from model import *
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

# 构造数据集（训练集、验证集）
train_data = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 获取数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度是：{}".format(train_data_size))
print("测试数据集的长度是：{}".format(test_data_size))

# 通过 DataLoader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 实例化模型
m = M()
if torch.cuda.is_available():
    m = m.cuda()

# 构造损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 构造优化器
learning_rate = 0.01  # 训练速率
optimizer = torch.optim.SGD(m.parameters(), lr=learning_rate)

print("----")
print(m.parameters())

# tensorborad
writer = SummaryWriter("logs")

# 训练参数
total_train_step = 0  # 训练次数
total_test_step = 0  # 测试次数
epoch = 20  # 训练轮数

for i in range(epoch):
    # 计时
    start_time = time.time()

    # 训练
    m.train()
    print("---------- 第 {} 轮训练开始 ----------".format(i + 1))

    # 训练开始
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = m(imgs)
        loss = loss_fn(outputs, targets)

        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        print("训练次数： {} ，Loss： {} ".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    m.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = m(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    total_test_step = total_test_step + 1
    print("整体测试集上的 Loss： {}".format(total_test_loss))
    print("整体测试集上的正确率： {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    # 统计当前轮次训练时间
    print("第 {} 轮耗时： {}".format(i, time.time() - start_time))

    # 保存每一轮训练的模型
    if epoch - i < 10:
        torch.save(m, "models/m_{}.pth".format(i))
        print("模型 m_{}.pth 已保存".format(i))

writer.close()

# pytorch 1.1.0
# GTX 1050 ti for once: 17.298532724380493
# i7-7700 HQ for once: 202.69984340667725

# pytorch 1.11.0 without tensorboard
# GTX 1050 ti for once: 18.37849235534668
# i7-7700 HQ for once: 54.89324951171875

# pytorch 1.7.0
# GTX 1050 ti for once: 20.135103225708008
# i7-7700 HQ for once: 73.6921615600586
