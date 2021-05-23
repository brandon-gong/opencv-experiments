#!/usr/bin/env python3

import numpy as np
from rubik_solver import utils
import cv2

# Normally, all you need is `cap = cv2.VideoCapture(0)`,
# But with this USB Logitech B525 webcam I had to mess around with
# a lot of properties to get a smooth video input.
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
W, H = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
# FINISHED INITIALIZING CAPTURE

# squaresstring = "".join(squares)
#
# # solve with red facing me, yellow facing up.
# print(utils.solve(squaresstring, 'Kociemba'))

while(True):
    ret, frame = cap.read()
    frame = frame[0:1080, 420:1500]

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)[1]
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((9,9),np.uint8))
    binary = cv2.dilate(binary,np.ones((9,9), np.uint8),iterations = 1)

    edges = cv2.Canny(binary,50,150,apertureSize = 3)
    cv2.imshow('frame',binary)


    #cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleaning up everything
cap.release()
cv2.destroyAllWindows()

