# -*-coding:utf-8 -*-
# File       : test.py
# Time       : 2022/5/11 13:51
# Author     : pangyafei
# Description:

import torch
import torchvision
from PIL import Image
from model import *

# 加载数据
image_path = 'data/pic/2.png'
image = Image.open(image_path)
image = image.convert("RGB")
print(image)

# resize
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)
image = torch.reshape(image, (1, 3, 32, 32))

# 加载模型
model = torch.load("m_19.pth")

model.eval()
with torch.no_grad():
    output = model(image)

print(output)

# 推理类型映射
dicts = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# 打印推理结果
print("图片分类结果：{}".format(dicts.get(output.argmax().item())))
