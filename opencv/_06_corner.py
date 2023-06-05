# -*-coding:utf-8 -*-
# File       : _06_corner.py
# Time       : 2022/5/18 9:38
# Author     : pangyafei
# Description: 图像特征点提取

import cv2

image = cv2.imread('../imgs/opencv_logo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 500, 0.1, 10)
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), 3)

cv2.imshow('image', image)
cv2.waitKey()
