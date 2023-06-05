# -*-coding:utf-8 -*-
# File       : _03_crop.py
# Time       : 2022/5/18 9:24
# Author     : pangyafei
# Description: 图像裁剪

import cv2

image = cv2.imread("../imgs/opencv_logo.jpg")

crop = image[10:170, 40:200]  # row, col
cv2.imshow('crop', crop)
cv2.waitKey()
