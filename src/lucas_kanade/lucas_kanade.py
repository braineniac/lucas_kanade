#!/usr/bin/env python3

import cv2
import numpy as np


def calcOpticalFlowPyrLK(prevImg, nextImg,
                         prevPts, nextPts=[],
                         status=[], er=[],
                         winSize=(3, 3), maxLevel=5,
                         criteria=[], flags=[], minEigThreshold=0):
    # get spacial derivatives
    Ix = cv2.Sobel(prevImg, -1, 1, 0)
    Iy = cv2.Sobel(prevImg, -1, 0, 1)

    It = np.abs(prevImg, nextImg)

    # create window and fill it
    Ix_win = np.zeros(winSize[0] * winSize[1])
    Iy_win = np.zeros(winSize[0] * winSize[1])
    It_win = np.zeros(winSize[0] * winSize[1])

    for point in prevPts:

        # remove outer list
        point = point[0]

        x, y = point
        x, y = int(x), int(y)

        # copy neighborhood points to windows
        dx_start = winSize[1]//2
        dy_start = winSize[0]//2
        dx_end = (winSize[1]//2) + 1
        dy_end = (winSize[0]//2) + 1

        Ix_win = Ix[
            (x-dx_start):(x+dx_end),
            (y-dy_start):(y+dy_end)].reshape((winSize[0] * winSize[1], 1))
        Iy_win = Iy[
            (x-dx_start):(x+dx_end),
            (y-dy_start):(y+dy_end)].reshape((winSize[0] * winSize[1], 1))
        It_win = It[
            (x-dx_start):(x+dx_end),
            (y-dy_start):(y+dy_end)].reshape((winSize[0] * winSize[1], 1))

        next_points = calcNextPts(Ix_win, Iy_win, It_win)
        print(next_points)

        exit(0)


def calcNextPts(Ix_win, Iy_win, It_win):
    # computes with least squares method y=Sp
    y = np.array(-It_win)
    S = np.concatenate([Ix_win, Iy_win], axis=1)
    x, res, rank, s = np.linalg.lstsq(S, y)
    return x


def buildOpticalFlowPyramid(img, pyramid, winSize,
                            maxLevel, withDerivatives,
                            pyrBorder, derivBorder,
                            tryResuseInputImage):
    pass
