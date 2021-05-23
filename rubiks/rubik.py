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

# final video outputs
originalframes = []
grayframes = []
binframes = []
edgeframes = []
minrectframes = []
endsolframes = []

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

        originalframes.append(resized.copy())

        # OpenCV's built-in grayscale takes the average of all the pixel
        # colors, which is quite bad for differentiating between black and
        # the red and blue color stickers on the cube.
        # So instead, I take the grayscale via the max color channel,
        # which gives brighter values for the darker red and blue stickers
        (b, g, r) = cv2.split(resized);
        gray = np.maximum(np.maximum(r, g), b)
        grayframes.append(gray.copy())

        binary = cv2.threshold(gray,130,255,cv2.THRESH_BINARY)[1]
        binframes.append(binary.copy())
        #binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((10,10),np.uint8))
        canny = cv2.Canny(binary, 127, 255, 1)
        edgeframes.append(canny.copy())

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

        minrectframes.append(resized.copy())

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

    cv2.imshow("asdf", resized)

squaresstring = "".join(squares)
cv2.destroyAllWindows()
presol = utils.solve(squaresstring, 'Kociemba')
print(presol)
solution = ["nd"]
for p in presol:
    p = p.raw
    factor = 1
    if p[-1] == "2":
        p = p[:-1]
        factor = 2
    solution += [p] * factor
solution += 2*["nd"]

rrects = []
while len(solution) != 0:
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


    if len(finalRects) == 9:
        rectAngles.sort()
        med = rectAngles[len(rectAngles) // 2]
        rects = []
        for r in finalRects:
            rects.append(r)

        if len(rects) != 9:
            # tried our best, discard
            #print("too many", len(rects))
            continue

        rects = sorted(rects, key = lambda r: r[0][1])
        rects[0:3] = sorted(rects[0:3], key = lambda r: r[0][0])
        rects[3:6] = sorted(rects[3:6], key = lambda r: r[0][0])
        rects[6:9] = sorted(rects[6:9], key = lambda r: r[0][0])

        rects = [(int(r[0][0]), int(r[0][1])) for r in rects]
        arrowsize = 10
        if solution[0] == 'L':
            resized = cv2.line(resized, rects[2], rects[8], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[2], (rects[2][0] + arrowsize, rects[2][1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[2], (rects[2][0] - arrowsize, rects[2][1] + arrowsize), (0, 255, 0), 5)
        elif solution[0] == "R'":
            resized = cv2.line(resized, rects[0], rects[6], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[0], (rects[0][0] + arrowsize, rects[0][1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[0], (rects[0][0] - arrowsize, rects[0][1] + arrowsize), (0, 255, 0), 5)
        elif solution[0] == "L'":
            resized = cv2.line(resized, rects[2], rects[8], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[8], (rects[8][0] - arrowsize, rects[8][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[8], (rects[8][0] + arrowsize, rects[8][1] - arrowsize), (0, 255, 0), 5)
        elif solution[0] == "R":
            resized = cv2.line(resized, rects[0], rects[6], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[6], (rects[6][0] - arrowsize, rects[6][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[6], (rects[6][0] + arrowsize, rects[6][1] - arrowsize), (0, 255, 0), 5)
        elif solution[0] == "F'":
            resized = cv2.line(resized, rects[0], rects[2], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[2], (rects[2][0] - arrowsize, rects[2][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[2], (rects[2][0] - arrowsize, rects[2][1] + arrowsize), (0, 255, 0), 5)
        elif solution[0] == "F":
            resized = cv2.line(resized, rects[0], rects[2], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[0], (rects[0][0] + arrowsize, rects[0][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[0], (rects[0][0] + arrowsize, rects[0][1] + arrowsize), (0, 255, 0), 5)
        elif solution[0] == "B":
            resized = cv2.line(resized, rects[6], rects[8], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[8], (rects[8][0] - arrowsize, rects[8][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[8], (rects[8][0] - arrowsize, rects[8][1] + arrowsize), (0, 255, 0), 5)
        elif solution[0] == "B'":
            resized = cv2.line(resized, rects[6], rects[8], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[6], (rects[6][0] + arrowsize, rects[6][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[6], (rects[6][0] + arrowsize, rects[6][1] + arrowsize), (0, 255, 0), 5)
        elif solution[0] == "U'":
            resized = cv2.line(resized, rects[6], rects[8], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[2], rects[8], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[0], rects[6], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[0], rects[2], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[1], (rects[1][0] + arrowsize, rects[1][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[1], (rects[1][0] + arrowsize, rects[1][1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[7], (rects[7][0] - arrowsize, rects[7][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[7], (rects[7][0] - arrowsize, rects[7][1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[3], (rects[3][0] + arrowsize, rects[3][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[3], (rects[3][0] - arrowsize, rects[3][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[5], (rects[5][0] - arrowsize, rects[5][1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[5], (rects[5][0] + arrowsize, rects[5][1] + arrowsize), (0, 255, 0), 5)
        elif solution[0] == "U":
            resized = cv2.line(resized, rects[6], rects[8], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[2], rects[8], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[0], rects[6], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[0], rects[2], (0, 255, 0), 5)
            resized = cv2.line(resized, rects[7], (rects[7][0] + arrowsize, rects[7][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[7], (rects[7][0] + arrowsize, rects[7][1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[1], (rects[1][0] - arrowsize, rects[1][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[1], (rects[1][0] - arrowsize, rects[1][1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[5], (rects[5][0] + arrowsize, rects[5][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[5], (rects[5][0] - arrowsize, rects[5][1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[3], (rects[3][0] - arrowsize, rects[3][1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rects[3], (rects[3][0] + arrowsize, rects[3][1] + arrowsize), (0, 255, 0), 5)
        elif solution[0] == "D":
            ul = (rects[0][0] - (rects[4][0] - rects[0][0]), rects[0][1] - (rects[4][1] - rects[0][1]))
            ur = (rects[2][0] + (rects[2][0] - rects[4][0]), rects[2][1] - (rects[4][1] - rects[2][1]))
            ll = (rects[6][0] + (rects[6][0] - rects[4][0]), rects[6][1] - (rects[4][1] - rects[6][1]))
            lr = (rects[8][0] + (rects[8][0] - rects[4][0]), rects[8][1] - (rects[4][1] - rects[8][1]))
            resized = cv2.line(resized, ul, ur, (0, 255, 0), 5)
            resized = cv2.line(resized, ul, ll, (0, 255, 0), 5)
            resized = cv2.line(resized, ur, lr, (0, 255, 0), 5)
            resized = cv2.line(resized, ll, lr, (0, 255, 0), 5)
            umed = ((ul[0]+ur[0])//2, (ul[1]+ur[1])//2)
            rmed = ((ur[0]+lr[0])//2, (ur[1]+lr[1])//2)
            lmed = ((ul[0]+ll[0])//2, (ul[1]+ll[1])//2)
            bmed = ((ll[0]+lr[0])//2, (ll[1]+lr[1])//2)
            resized = cv2.line(resized, umed, (umed[0] + arrowsize, umed[1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, umed, (umed[0] + arrowsize, umed[1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, bmed, (bmed[0] - arrowsize, bmed[1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, bmed, (bmed[0] - arrowsize, bmed[1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, lmed, (lmed[0] + arrowsize, lmed[1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, lmed, (lmed[0] - arrowsize, lmed[1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rmed, (rmed[0] - arrowsize, rmed[1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rmed, (rmed[0] + arrowsize, rmed[1] + arrowsize), (0, 255, 0), 5)
        elif solution[0] == "D'":
            ul = (rects[0][0] - (rects[4][0] - rects[0][0]), rects[0][1] - (rects[4][1] - rects[0][1]))
            ur = (rects[2][0] + (rects[2][0] - rects[4][0]), rects[2][1] - (rects[4][1] - rects[2][1]))
            ll = (rects[6][0] + (rects[6][0] - rects[4][0]), rects[6][1] - (rects[4][1] - rects[6][1]))
            lr = (rects[8][0] + (rects[8][0] - rects[4][0]), rects[8][1] - (rects[4][1] - rects[8][1]))
            resized = cv2.line(resized, ul, ur, (0, 255, 0), 5)
            resized = cv2.line(resized, ul, ll, (0, 255, 0), 5)
            resized = cv2.line(resized, ur, lr, (0, 255, 0), 5)
            resized = cv2.line(resized, ll, lr, (0, 255, 0), 5)
            umed = ((ul[0]+ur[0])//2, (ul[1]+ur[1])//2)
            rmed = ((ur[0]+lr[0])//2, (ur[1]+lr[1])//2)
            lmed = ((ul[0]+ll[0])//2, (ul[1]+ll[1])//2)
            bmed = ((ll[0]+lr[0])//2, (ll[1]+lr[1])//2)
            resized = cv2.line(resized, bmed, (bmed[0] + arrowsize, bmed[1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, bmed, (bmed[0] + arrowsize, bmed[1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, umed, (umed[0] - arrowsize, umed[1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, umed, (umed[0] - arrowsize, umed[1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rmed, (rmed[0] + arrowsize, rmed[1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, rmed, (rmed[0] - arrowsize, rmed[1] - arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, lmed, (lmed[0] - arrowsize, lmed[1] + arrowsize), (0, 255, 0), 5)
            resized = cv2.line(resized, lmed, (lmed[0] + arrowsize, lmed[1] + arrowsize), (0, 255, 0), 5)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        print("here")
        solution.pop(0)




    # Display the resulting frame
    cv2.imshow('frame', resized)
    endsolframes.append(resized.copy())

#originalframes = []
#grayframes = []
#binframes = []
#edgeframes = []
#minrectframes = []
#endsolframes = []

print("writing original...")
originalout = cv2.VideoWriter('original.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1080,1080))
for frame in originalframes:
    originalout.write(frame)
originalout.release()
print("writing gray...")
grayout = cv2.VideoWriter('gray.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1080,1080))
for frame in grayframes:
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    grayout.write(frame)
grayout.release()
print("writing bin...")
binout = cv2.VideoWriter('bin.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1080,1080))
for frame in binframes:
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    binout.write(frame)
binout.release()
print("writing edge...")
edgeout = cv2.VideoWriter('edge.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1080,1080))
for frame in edgeframes:
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    edgeout.write(frame)
edgeout.release()
print("writing minrect...")
minrectout = cv2.VideoWriter('minrect.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1080,1080))
for frame in minrectframes:
    minrectout.write(frame)
minrectout.release()
print("writing endsol...")
endsolout = cv2.VideoWriter('endsol.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1080,1080))
for frame in endsolframes:
    endsolout.write(frame)
endsolout.release()


# cleaning up everything
cap.release()
cv2.destroyAllWindows()

