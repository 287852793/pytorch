# -*-coding:utf-8 -*-
# File       : _09_threshold.py
# Time       : 2022/5/18 10:50
# Author     : pangyafei
# Description: 阈值算法（二值化算法）

import cv2

gray = cv2.imread('../imgs/bookpage.jpg', cv2.IMREAD_GRAYSCALE)
ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # 自定义参数算法：阈值 10， 最大灰度255
binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)  # 自适应阈值算法
ret2, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # otsu算法，最大类间算法，大津算法
# print(ret)

cv2.imshow('gray', gray)
cv2.imshow('binary', binary)
cv2.imshow('adaptive', binary_adaptive)
cv2.imshow('binary2', binary2)
cv2.waitKey()
