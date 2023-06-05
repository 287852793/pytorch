# -*-coding:utf-8 -*-
# File       : _10_morphology.py
# Time       : 2022/5/18 11:01
# Author     : pangyafei
# Description: 图像形态学算法(腐蚀、膨胀）,清理图像边缘细节

import cv2
import numpy as np

gray = cv2.imread('../imgs/opencv_logo.jpg', cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
kernal = np.ones((5, 5), np.uint8)  # 核大小
erosion = cv2.erode(binary, kernal)  # 腐蚀算法
dilation = cv2.dilate(binary, kernal)  # 膨胀算法


cv2.imshow('binary', binary)
cv2.imshow('erosion', erosion)
cv2.imshow('dilation', dilation)

binary = cv2.dilate(binary, kernal)
binary = cv2.erode(binary, kernal)
binary = cv2.dilate(binary, kernal)
binary = cv2.erode(binary, kernal)
binary = cv2.dilate(binary, kernal)
binary = cv2.erode(binary, kernal)
binary = cv2.dilate(binary, kernal)
binary = cv2.erode(binary, kernal)

cv2.imshow('binary2', binary)

cv2.waitKey()
