#!/usr/bin/env python3

"""
Routine to read current Rubiks cube state from webcam,
i.e. getting colors of each square and putting it in a matrix
to solve later.

Author: Brandon Gong
Date: 4/20/2021
"""

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

squares = ['_'] * 54
while('_' in squares):
    faces = ['_'] * 9
    while('_' in faces):
        faces = ['_'] * 9
        # read next frame from stream
        ret, frame = cap.read()
        frame = frame[0:1080, 420:1500]

        scale = 1
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dim = (width, height)

        # resize the image down - 1920x1080 is way too much extra info.
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        # OpenCV's built-in grayscale takes the average of all the pixel
        # colors, which is quite bad for differentiating between black and
        # the red and blue color stickers on the cube.
        # So instead, I take the grayscale via the max color channel,
        # which gives brighter values for the darker red and blue stickers
        (b, g, r) = cv2.split(resized);
        gray = np.maximum(np.maximum(r, g), b)

        binary = cv2.threshold(gray,140,255,cv2.THRESH_BINARY)[1]
        canny = cv2.Canny(binary, 127, 255, 1)

        cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        rects = [cv2.minAreaRect(c) for c in cnts]

        finalRects = []
        rectAngles = []

        for r in rects:

            if r[1][0] * r[1][1] < 2000 or r[1][0] * r[1][1] > 6000:
                continue
            if r[1][0] / r[1][1] >= 1.333 or r[1][0] / r[1][1] <= 0.75:
                continue
            finalRects.append(r)
            rectAngles.append(r[2])
            cv2.drawContours(resized, [np.int0(cv2.boxPoints(r))], 0, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('frame', resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(finalRects) < 9:
            print("not enough")
            continue

        rectAngles.sort()
        med = rectAngles[len(rectAngles) // 2]
        rects = []
        for r in finalRects:
            rects.append(r)

        if len(rects) != 9:
            # tried our best, discard
            print("too many", len(rects))
            continue

        rects = sorted(rects, key = lambda r: r[0][1])
        rects[0:3] = sorted(rects[0:3], key = lambda r: r[0][0])
        rects[3:6] = sorted(rects[3:6], key = lambda r: r[0][0])
        rects[6:9] = sorted(rects[6:9], key = lambda r: r[0][0])

        colors = []
        for r in rects:
            color = resized[int(r[0][1])][int(r[0][0])]
            colors.append(cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0])

        for i,c in enumerate(colors):
            if c[2] > 200 and c[1] < 20:
                faces[i] = "w"
            elif c[0] > 30 and c[0] < 39 and c[2] > 230:
                faces[i] = "y"
            elif c[0] > 100 and c[0] < 112 and c[2] > 170:
                faces[i] = "b"
            elif c[0] < 10 and c[2] > 230:
                faces[i] = "o"
            elif c[0] > 72 and c[0] < 82 and c[2] > 155:
                faces[i] = "g"
            elif (c[0] < 5 or c[0] > 175):
                faces[i] = "r"
            elif i == 4:
                faces[i] = "w"
            else:
                faces[i] = "_"

        cv2.imshow('asdf', resized)

    #                ----------------
    #                | 0  | 1  | 2  |
    #                ----------------
    #                | 3  | y. | 5  |
    #                ----------------
    #                | 6  | 7  | 8  |
    #                ----------------
    # -------------------------------------------------------------
    # | 9  | 10 | 11 | 18 | 19 | 20 | 27 | 28 | 29 | 36 | 37 | 38 |
    # -------------------------------------------------------------
    # | 12 | b. | 14 | 21 | r. | 23 | 30 | g. | 32 | 39 | o. | 41 |
    # -------------------------------------------------------------
    # | 15 | 16 | 17 | 24 | 25 | 26 | 33 | 34 | 35 | 42 | 43 | 44 |
    # -------------------------------------------------------------
    #                ----------------
    #                | 45 | 46 | 47 |
    #                ----------------
    #                | 48 | w. | 50 |
    #                ----------------
    #                | 51 | 52 | 53 |
    #                ----------------
    # standard orientation: white facing up, orange facing me, red facing lamp
    if faces[4] == "w":
        for i in range(9):
            squares[i + 45] = faces[8-i]
    elif faces[4] == "r":
        for i in range(9):
            squares[i + 18] = faces[8-i]
    elif faces[4] == "y":
        for i in range(9):
            squares[i] = faces[8 - i]
    elif faces[4] == "o":
        for i in range(9):
            squares[i + 36] = faces[i]
    elif faces[4] == "b":
        for i in range(9):
            squares[i + 9] = faces[i]
    elif faces[4] == "g":
        for i in range(9):
            squares[i+27] = faces[i]

squaresstring = "".join(squares)

# solve with red facing me, yellow facing up.
print(utils.solve(squaresstring, 'Kociemba'))

# cleaning up everything
cap.release()
cv2.destroyAllWindows()

