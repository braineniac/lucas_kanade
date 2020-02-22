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
    I = (Ix, Iy, It)

    # create neighborhood vecotors
    Ix_v = np.zeros(winSize[0] * winSize[1])
    Iy_v = np.zeros(winSize[0] * winSize[1])
    It_v = np.zeros(winSize[0] * winSize[1])
    I_v = (Ix_v, Iy_v, It_v)

    # window border indexes
    dx_start = winSize[1]//2
    dy_start = winSize[0]//2
    dx_end = (winSize[1]//2) + 1
    dy_end = (winSize[0]//2) + 1
    win_ind = ((dx_start, dx_end), (dy_start, dy_end))

    st = np.zeros((prevPts.shape[0], 1), dtype=bool)  # status
    nextPts = np.zeros(prevPts.shape, dtype=int)
    err = []

    for i in range(0, prevPts.shape[0]):
        nextPt, status = calcPointFlow(I, I_v, prevPts[i], win_ind)
        nextPts[i] = nextPt
        st[i] = status

    return nextPts, st, err


def calcPointFlow(I, I_v, prevPt, win_ind):
    Ix, Iy, It = I
    Ix_v, Iy_v, It_v = I_v

    x, y = prevPt[0]  # remove outer list
    x, y = int(x), int(y)

    # point neighborhood indexes
    dx, dy = win_ind
    dx_start, dx_end = dx
    dy_start, dy_end = dy
    num_win_elem = (dx_end + dx_start) * (dy_end + dy_start)

    try:
        # copy neighborhood points to column vectors
        Ix_v = Ix[
            (x-dx_start):(x+dx_end),
            (y-dy_start):(y+dy_end)].reshape((num_win_elem, 1))
        Iy_v = Iy[
            (x-dx_start):(x+dx_end),
            (y-dy_start):(y+dy_end)].reshape((num_win_elem, 1))
        It_v = It[
            (x-dx_start):(x+dx_end),
            (y-dy_start):(y+dy_end)].reshape((num_win_elem, 1))

        uv = calcNextPt(Ix_v, Iy_v, It_v).reshape((1, 2))
        nextPt = prevPt + uv
        nextPt[0][0] = np.rint(nextPt[0][0])
        nextPt[0][1] = np.rint(nextPt[0][1])
        return nextPt, True
    except ValueError:
        # in this case there is no neighborhood
        return np.nan, False


def calcNextPt(Ix_v, Iy_v, It_v):
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
