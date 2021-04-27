#! /usr/bin/env python3

import numpy as np
import random
import cv2

# render a go board into a matrix.
# state should be a string of 9x9, 13x13, 19x19.
# size is the dimensions of the resulting image.
def render_board(state, size=500):
    margin = size // 10
    r = np.zeros((size,size,3), np.uint8)
    r[:] = (163, 218, 255)
    dims = 19
    if len(state) == 81:
        dims = 9
    elif len(state) == 169:
        dims = 13

    grid_size = (size - 2*margin) / (dims-1)

    for i in range(dims):
        r = cv2.line(r, (int(margin + i*grid_size), margin), (int(margin + i * grid_size), size - margin), (0, 0, 0), 1)
    for i in range(dims):
        r = cv2.line(r, (margin, int(margin + i*grid_size)), (size - margin, int(margin + i*grid_size)), (0, 0, 0), 1)

    # for 9x9, there is a star point in the middle
    if dims == 9:
        r = cv2.circle(r, (int(margin + 4 * grid_size), int(margin + 4*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
    if dims == 13:
        r = cv2.circle(r, (int(margin + 3 * grid_size), int(margin + 3*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 9 * grid_size), int(margin + 9*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 3 * grid_size), int(margin + 9*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 9 * grid_size), int(margin + 3*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 6 * grid_size), int(margin + 6*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
    if dims == 19:
        r = cv2.circle(r, (int(margin + 3 * grid_size), int(margin + 3*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 15 * grid_size), int(margin + 15*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 3 * grid_size), int(margin + 15*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 15 * grid_size), int(margin + 3*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 9 * grid_size), int(margin + 9 * grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 9 * grid_size), int(margin + 3*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 9 * grid_size), int(margin + 15*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 3 * grid_size), int(margin + 9*grid_size)), int(grid_size / 8), (0, 0, 0), -1)
        r = cv2.circle(r, (int(margin + 15 * grid_size), int(margin + 9*grid_size)), int(grid_size / 8), (0, 0, 0), -1)

    # iterate through state, w is a white stone, b is a black stone, and anything else is blank
    for i, c in enumerate(state):
        gx = i % dims
        gy = i // dims
        if c == 'w':
            r = cv2.circle(r, (int(margin + gx * grid_size), int(margin + gy * grid_size)), int(grid_size * 0.45), (255, 255, 255), -1)
            r = cv2.circle(r, (int(margin + gx * grid_size), int(margin + gy * grid_size)), int(grid_size * 0.45), (0, 0, 0), 1)
        elif c == 'b':
            r = cv2.circle(r, (int(margin + gx * grid_size), int(margin + gy * grid_size)), int(grid_size * 0.45), (0, 0, 0), -1)

    return r


if __name__ == "__main__":
    while True:
        inp = ""
        for i in range(random.choice((9*9, 13*13, 19*19))):
            inp += random.choice(("b", "w", "_"))
        x = render_board(inp, size=1000)
        cv2.imshow("board", x)
        while True:
            if cv2.waitKey(0) & 0xFF == ord('n'):
                break
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                quit()

