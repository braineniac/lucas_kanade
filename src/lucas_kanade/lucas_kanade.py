#!/usr/bin/env python3

import cv2
import numpy as np


def calcOpticalFlowPyrLK(prevImg, nextImg,
                         prevPts, nextPts=[],
                         status=[], er=[],
                         winSize=(3, 3), maxLevel=5,
                         criteria=[], flags=[], minEigThreshold=0):
    # check window size
    if winSize[0] % 2 != 1 or winSize[1] % 2 != 1:
        print("winSize must be an odd number!")
        exit(-1)

    # get derivatives
    Ix = cv2.Sobel(prevImg, -1, 1, 0)
    Iy = cv2.Sobel(prevImg, -1, 0, 1)
    It = np.abs(prevImg, nextImg)

    # create neighborhood vecotors
    Ix_v = np.zeros(winSize[0] * winSize[1])
    Iy_v = np.zeros(winSize[0] * winSize[1])
    It_v = np.zeros(winSize[0] * winSize[1])

    st = np.zeros((prevPts.shape[0], 1), dtype=bool)  # status
    nextPts = np.zeros(prevPts.shape, dtype=int)
    err = []

    for i in range(0, prevPts.shape[0]):
        prevPt = prevPts[i][0]  # remove outer list
        x, y = prevPt
        x, y = int(x), int(y)

        dx_start = winSize[1]//2
        dy_start = winSize[0]//2
        dx_end = (winSize[1]//2) + 1
        dy_end = (winSize[0]//2) + 1

        try:
            # copy neighborhood points to column vectors
            Ix_v = Ix[
                (x-dx_start):(x+dx_end),
                (y-dy_start):(y+dy_end)].reshape((winSize[0] * winSize[1], 1))
            Iy_v = Iy[
                (x-dx_start):(x+dx_end),
                (y-dy_start):(y+dy_end)].reshape((winSize[0] * winSize[1], 1))
            It_v = It[
                (x-dx_start):(x+dx_end),
                (y-dy_start):(y+dy_end)].reshape((winSize[0] * winSize[1], 1))

            uv = calcNextPts(Ix_v, Iy_v, It_v).reshape((1, 2))
            nextPt = prevPt + uv
            nextPts[i][0][0] = np.rint(nextPt[0][0])
            nextPts[i][0][1] = np.rint(nextPt[0][1])
            st[i] = True  # point is good

        except ValueError:
            # in this case there is no neighborhood, skipping point
            pass

    # print(nextPts)
    # print(prevPts)
    # print(st)
    #print(nextPts)
    return nextPts, st, err


def calcNextPts(Ix_v, Iy_v, It_v):
    # computes with least squares method y=Sp
    y = np.array(-It_v)
    S = np.concatenate([Ix_v, Iy_v], axis=1)
    x, res, rank, s = np.linalg.lstsq(S, y)
    return x


def buildOpticalFlowPyramid(img, pyramid, winSize,
                            maxLevel, withDerivatives,
                            pyrBorder, derivBorder,
                            tryResuseInputImage):
    pass
