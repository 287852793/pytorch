# -*-coding:utf-8 -*-
# File       : _11_camera.py
# Time       : 2022/5/18 11:09
# Author     : pangyafei
# Description: opencv 调用电脑摄像头

import cv2

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    cv2.imshow('camera', frame)
    key = cv2.waitKey(1)
    if key != -1:
        break

capture.release()
