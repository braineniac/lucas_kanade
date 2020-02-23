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

    # create neighborhood vectors
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

    st = np.ones((prevPts.shape[0], 1), dtype=bool)  # status
    nextPts = np.zeros(prevPts.shape, dtype=int)
    err = []

    prev_pyramid = buildOpticalFlowPyramid(prevImg, maxLevel)
    next_pyramid = buildOpticalFlowPyramid(nextImg, maxLevel)

    # scale points for highest level pyramid
    scalePts(prevPts, 1/(2**maxLevel))

    for i in range(len(prev_pyramid)-1, -1, -1):
        # get derivatives
        Ix = cv2.Sobel(prev_pyramid[i], -1, 1, 0)
        Iy = cv2.Sobel(prev_pyramid[i], -1, 0, 1)
        It = np.abs(next_pyramid[i], next_pyramid[i])
        I = (Ix, Iy, It)

        for j in range(prevPts.shape[0]):
            nextPt, status = calcPointFlow(I, I_v, prevPts[j], st[j], win_ind)
            nextPts[j] = nextPt
            st[j] = status

        scalePts(nextPts, 2)
        prevPts = nextPts
        #print(prevPts)
    return nextPts, st, err


def scalePts(Pts, scale):
    for x in np.nditer(Pts, op_flags=['writeonly']):
        x[...] = np.rint(x * scale)


def calcPointFlow(I, I_v, prevPt, st, win_ind):
    if not st:
        return np.nan, False
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


def buildOpticalFlowPyramid(img, maxLevel, winSize=None,
                            pyramid=None, pyrBorder=cv2.INTER_NEAREST):
    pyramid = []
    pyramid.append(img)
    for i in range(maxLevel+1):
        pyramid.append(pyrDown(pyramid[i], borderType=pyrBorder))
    return pyramid


def pyrDown(img, out=None,
            outSize=None,
            borderType=cv2.INTER_NEAREST):
    if outSize is None:
        outSize = (img.shape[1]//2, img.shape[0]//2)

    # remove every even row and column
    img = np.delete(img, np.arange(0, img.shape[0], 2), 0)
    img = np.delete(img, np.arange(0, img.shape[1], 2), 1)

    # 5x5 gaussian kernel should be good enough
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0, 0)

    return blurred_img
