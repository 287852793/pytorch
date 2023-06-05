# -*-coding:utf-8 -*-
# File       : _01_hello.py
# Time       : 2022/5/18 9:16
# Author     : pangyafei
# Description: 图像加载

import cv2

print(cv2.getVersionString())

image = cv2.imread('../imgs/bookpage.jpg')
print(image.shape)

cv2.imshow('image', image)
cv2.waitKey()
