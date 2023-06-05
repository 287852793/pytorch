# -*-coding:utf-8 -*-
# File       : predict.py
# Time       : 2022/5/10 10:02
# Author     : pangyafei
# Description:

import torch

# 某一轮次的分类结果 outputs
outputs = torch.tensor([[0.1, 0.2],
                        [0.3, 0.4]])
targets = torch.tensor([0, 1])

print(outputs.argmax(1))
# 输出 tensor([1, 1])，代表两个样本分类为类型1
print((outputs.argmax(1) == targets).sum())
# 输出 tensor(1)，代表两个分类正确的总数为1个样本
print(((outputs.argmax(1) == targets).sum() / len(targets)).item())
# 输出 0.5000，代表分类准确率为 50%
