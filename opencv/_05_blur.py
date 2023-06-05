# -*-coding:utf-8 -*-
# File       : _05_blur.py
# Time       : 2022/5/18 9:33
# Author     : pangyafei
# Description: 图像滤波

import cv2

image = cv2.imread('../imgs/plane.jpg')

gauss = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯滤波
median = cv2.medianBlur(image, 5)  # 中值滤波

cv2.imshow('gauss', gauss)
cv2.imshow('median', median)
cv2.waitKey()
