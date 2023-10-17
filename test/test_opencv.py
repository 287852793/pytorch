#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/14 9:11
# @Author  : pyf
# @File    : test_opencv.py
# @Description : 测试屋顶结构线提取

import cv2
import numpy as np

# 加载房屋顶部图片
img = cv2.imread('imgs/building_0_5meter.jpg')

# 将图像转换为灰度图像
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape[0])
img = cv2.resize(img, dsize=(img.shape[1] * 10, img.shape[0] * 10), interpolation=cv2.INTER_LINEAR)
# cv2.imshow('image', img)
# cv2.waitKey(0)

img = cv2.medianBlur(img, 11)
cv2.imshow('image', img)

# 使用Canny边缘检测算法检测边缘
edges = cv2.Canny(img, 10, 20)

cv2.imshow('image0', edges)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilated = cv2.dilate(edges, kernel, iterations=3)
eroded = cv2.erode(dilated, kernel, iterations=3)
dilated = cv2.dilate(eroded, kernel, iterations=3)
eroded = cv2.erode(dilated, kernel, iterations=3)
dilated = cv2.dilate(eroded, kernel, iterations=3)
eroded = cv2.erode(dilated, kernel, iterations=3)
dilated = cv2.dilate(eroded, kernel, iterations=3)
eroded = cv2.erode(dilated, kernel, iterations=3)
dilated = cv2.dilate(eroded, kernel, iterations=3)
eroded = cv2.erode(dilated, kernel, iterations=3)
dilated = cv2.dilate(eroded, kernel, iterations=3)
eroded = cv2.erode(dilated, kernel, iterations=3)

cv2.imshow('image1', eroded)

# 查找轮廓
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

# t1 = cv2.GaussianBlur(image, (0, 0), 5)
# cv2.imshow('image1', t1)

t2 = cv2.medianBlur(image, 11)
cv2.imshow('image2', t2)

# t3 = cv2.bilateralFilter(t2, 9, 75, 75)
# cv2.imshow('image3', t3)

# 遍历每一个轮廓
# for contour in contours:
#     # 获取轮廓的边界框
#     x, y, w, h = cv2.boundingRect(contour)
#
#     # 画出边界框
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# binary = np.array(binary, dtype=np.uint8)
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# image = cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
# cv2.imshow('image1', image)

# # 使用Hough变换检测直线
# lines = cv2.HoughLinesP(edges, 10, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
# print(lines)
#
# # 在原图像上绘制检测到的直线
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果图像
# cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
