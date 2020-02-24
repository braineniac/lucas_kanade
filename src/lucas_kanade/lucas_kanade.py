#!/usr/bin/env python3

import cv2 as cv
import numpy as np


def calcOpticalFlowPyrLK(prevImg, nextImg,
                         prevPts, nextPts=None,
                         status=None, err=None,
                         winSize=(5, 5), maxLevel=3,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                         minEigThreshold=1):
    # check window size
    if winSize[0] % 2 != 1 or winSize[1] % 2 != 1:
        print("winSize must be an odd number!")
        exit(-1)

    # create neighborhood vectors
    Ix_v = np.zeros(winSize[0] * winSize[1])
    Iy_v = np.zeros(winSize[0] * winSize[1])
    It_v = np.zeros(winSize[0] * winSize[1])
    I_v = (Ix_v, Iy_v, It_v)

    st = np.ones((prevPts.shape[0], 1), dtype=bool)  # status
    nextPts = np.copy(prevPts)
    err = np.zeros((prevPts.shape[0], 1), dtype=float)

    prev_pyramid = buildOpticalFlowPyramid(prevImg, maxLevel)
    next_pyramid = buildOpticalFlowPyramid(nextImg, maxLevel)

    scalePts(nextPts, 1/(2**(maxLevel+1)))

    # iterate from highest to lowest level
    for i in range(len(prev_pyramid)-1, -1, -1):
        # get derivatives
        Ix = cv.Sobel(prev_pyramid[i], -1, 1, 0, 3)
        Iy = cv.Sobel(prev_pyramid[i], -1, 0, 1, 3)
        It = next_pyramid[i] - prev_pyramid[i]
        I = (Ix, Iy, It)

        # scale points for this pyramid level
        scalePts(nextPts, 2)

        for j in range(nextPts.shape[0]):
            k = 0
            while(k < criteria[1]):
                nextPt, status = calcPointFlow(I, I_v, nextPts[j], st[j], winSize, minEigThreshold)
                x_crit = nextPt[0][0] - prevPts[j][0][0] < criteria[2]
                y_crit = nextPt[0][1] - prevPts[j][0][1] < criteria[2]
                if x_crit and y_crit:
                    break
                nextPts[j] = nextPt
                st[j] = status
                k += 1
    return nextPts, st, err


def scalePts(Pts, scale):
    for x in np.nditer(Pts, op_flags=['writeonly']):
        x[...] = np.rint(x * scale)


def calcPointFlow(I, I_v, prevPt, st, winSize, minEigThreshold):
    if not st:
        return prevPt, False
    Ix, Iy, It = I
    Ix_v, Iy_v, It_v = I_v

    x, y = prevPt[0]  # remove outer list
    x, y = int(x), int(y)

    # point neighborhood indexes
    dx_start = winSize[1]//2
    dy_start = winSize[0]//2
    dx_end = (winSize[1]//2) + 1
    dy_end = (winSize[0]//2) + 1
    num_win_elem = winSize[0] * winSize[1]

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

        uv, res, s, eig = calcNextPt(Ix_v, Iy_v, It_v, winSize, minEigThreshold)
        uv = uv.reshape((1, 2))
        nextPt = prevPt + uv
        # print(nextPt, prevPt, uv, res/num_win_elem, s, eig)
        nextPt[0][0] = np.rint(nextPt[0][0])
        nextPt[0][1] = np.rint(nextPt[0][1])
        return nextPt, True
    except ValueError:
        # something went wrong with the calculation
        return prevPt, False
    except np.linalg.LinAlgError:
        # calculation didn't converge
        return prevPt, False


def calcNextPt(Ix_v, Iy_v, It_v, winSize, minEigThreshold):
    # computes with least squares method y=Sp
    y = np.array(-It_v)
    S = np.concatenate([Ix_v, Iy_v], axis=1)

    # get rid of badly conditioned points
    eig = np.linalg.eigvals(S.T.dot(S))
    if eig[0] < minEigThreshold or eig[1] < minEigThreshold:
        raise ValueError

    x, res, rank, s = np.linalg.lstsq(S, y, rcond=-1)

    # filter out points that are undetectably far away
    if x[0] > winSize[0]//2 or x[1] > winSize[1]//2:
        raise ValueError
    return x, res, s, eig


def buildOpticalFlowPyramid(img, maxLevel, winSize=None, pyramid=None):
    if pyramid is None:
        pyramid = []
    pyramid.append(img)
    for i in range(maxLevel):
        pyramid.append(pyrDown(pyramid[i]))
    return pyramid


def pyrDown(img, out=None, outSize=None):
    if outSize is None:
        outSize = (img.shape[1]//2, img.shape[0]//2)

    # 5x5 gaussian kernel should be good enough
    img = cv.GaussianBlur(img, (5, 5), 0, 0)

    # remove every even row and column
    img = np.delete(img, np.arange(0, img.shape[0], 2), 0)
    img = np.delete(img, np.arange(0, img.shape[1], 2), 1)

    return img
