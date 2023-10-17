#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 9:29
# @Author  : pyf
# @File    : test_tensor.py
# @Description : test tensor

import torch

# 随机初始化正态分布矩阵
x = torch.randn(2, 3)
print(x)

# 随机初始化均匀分布矩阵
x0 = torch.rand(2, 3)
print(x0)

# 未初始化的矩阵
x1 = torch.empty(2, 3)
print(x1)

# 全零矩阵,整型
x2 = torch.zeros(2, 3, dtype=torch.int)
print('x2', x2)

# 全零矩阵，浮点型
x3 = torch.zeros(2, 3)
print('x2', x3)

# 返回一个填充了随机整数的张量，这些整数在low和high之间均匀生成。张量的shape由变量参数size定义。
x16 = torch.randint(low=0, high=3, size=(10,)).float()
print("x16", x16)
print("x16.size()", x16.size())

# x2和x3沿着维度0（行）拼接
y = torch.cat((x2, x3), 0)
print("x2和x3拼接", y)

# 将数组转换为tensor型
x4 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('x4', x4)

# 获取tensor的维度
shape = x4.size()
print(shape)

x5 = x4.view(3, 2)  # 对tensor形状进行改变
print('x5', x5)

# -1代表系统自动补齐对应维数。2x3=6xwhat
x6 = x4.view(2, 1, -1)
print('x6', x6)

# 使用add函数会生成一个新的Tensor变量
x7 = x4.add(10)
print('x4', x4)
print('x4', x7)

# add_ 函数会直接再当前Tensor变量上进行操作。所以，对于函数名末尾带有"_"的函数都是会对Tensor变量本身进行操作的。可以直接加到x4上面。
x4.add_(10)
print('x4', x4)

# 矩阵乘法,.t()表示矩阵转置
x8 = torch.mm(x4, x4.t())
print('x8', x8)

# 对应元素相乘
x9 = x4 * x4
print(x9)

# tensor转换为numpy数组
x10 = x9.numpy()
print(x10)

# x9,x10共享同个内存，修改会同时修改
x9.add_(1)
print(x9)
print(x10)

# 将numpy数组转化为tensor
x11 = torch.from_numpy(x10)
print(x11)

print('=====')

x12 = torch.rand(4, 8, 6)
# print(x12)
# dim：维度，按照给定的维度去分，第二个参数如果是 int，则分成两个块，且第一个块的size 等于这个 int， 必须够分
# 第二个块的size 小于等于这个int，不能大于
# 如果是 list，则按list 给定的大小分成 list.size 个块
y1, y2 = torch.split(x12, 6, dim=1)
print(y1)
print(y2)

# 生成一个一维的矩阵
x13 = torch.randn(3)
print('x13', x13)
# 求平均值
x14 = torch.mean(x13)
print('x14', x14)

# 对每一分量求平方
x15 = torch.pow(x13, 2)
print('x15', x15)

# 求平均值
a = torch.randn(4, 4)
print(a)
# 沿着维度0（行）求均值
c = torch.mean(a, dim=0, keepdim=True)
print(c)
# 沿着维度1（列）求均值
d = torch.mean(a, dim=1, keepdim=True)
print(d)
# 求整个二维张量的平均值（两种形式的结果在结构上有区别）
e = torch.mean(a, dim=[0, 1], keepdim=True)
f = torch.mean(a)
print(e)
print(f)

# 创建一维整型tensor，并不包含end
z = torch.arange(0, 6)
# 改变tensor形状
p = z.view(2, 3)
print(p)
# 在第零维增加一个维度
q = p.unsqueeze(0)
print(q)
print(q.size())

# 去掉第零维度（只能去掉维度为1的维度）
t = q.squeeze(0)
print(t)
print(t.size())

print('=====')

a = torch.Tensor([[0, 0, 0], [-1, 1, 0], [2, 3, 4]])
print(a)
b = a[:, 0]  # 取二维tensor中第1维（列）第0列的所有数据。
print(b)
c = a[:, 2]  # 取二维tensor中第1维（列）第2列的所有数据。
print(c)
d = a[:, 0::2]  # 取二维tensor第1维（列）第0列到第1列的所有数据（不含末尾）。
print(d)
print(a[1::2])
