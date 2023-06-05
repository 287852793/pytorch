# -*-coding:utf-8 -*-
# File       : _08_gradient.py
# Time       : 2022/5/18 10:39
# Author     : pangyafei
# Description: 梯度算法

import cv2

gray = cv2.imread('../imgs/opencv_logo.jpg', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('gray', gray)
# cv2.waitKey()

laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # 梯度算法
canny = cv2.Canny(gray, 100, 200)  # 边缘检测算法

cv2.imshow('laplacian', laplacian)
cv2.imshow('canny', canny)
cv2.waitKey()
