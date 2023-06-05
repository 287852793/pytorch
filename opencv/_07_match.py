# -*-coding:utf-8 -*-
# File       : _07_match.py
# Time       : 2022/5/18 9:46
# Author     : pangyafei
# Description: 图像模板匹配

import cv2
import numpy as np

image = cv2.imread('../imgs/poker.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

template = gray[69:105, 234:264]  # 模板

# cv2.rectangle(gray, (234, 69), (264, 105), (0, 0, 255), 1)
# cv2.imshow('gray', template)
# cv2.waitKey()

match = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
locations = np.where(match >= 0.9)  # 匹配度大于90%
print(locations)

h, w = template.shape[0:2]

for p in zip(*locations[::-1]):
    x1, y1 = p[0], p[1]
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow('image', image)
cv2.waitKey()
