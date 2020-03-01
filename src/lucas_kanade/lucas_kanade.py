#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import multiprocessing as mp
from multiprocessing.dummy import Pool as Threadpool


def calcOpticalFlowPyrLK(prevImg, nextImg,
                         prevPts, nextPts=None,
                         status=None, err=None,
                         winSize=(5, 5), maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_COUNT, 10),
                         flags=[],
                         minEigThreshold=None):

    # check window size
    if winSize[0] % 2 != 1 or winSize[1] % 2 != 1:
        print("winSize must be an odd number!")
        exit(-1)

    # check for flags
    if cv.OPTFLOW_USE_INITIAL_FLOW not in flags:
        nextPts = np.copy(prevPts)

    # set error type
    err_type = 0
    if cv.OPTFLOW_LK_GET_MIN_EIGENVALS in flags:
        err_type = cv.OPTFLOW_LK_GET_MIN_EIGENVALS

    # criteria check
    if (criteria[0] != cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT
            and criteria[0] != cv.TERM_CRITERIA_EPS
            and criteria[0] != cv.TERM_CRITERIA_COUNT):
        print("Wrong criteria passed!")
        exit(-1)

    if status is None:
        status = np.ones((prevPts.shape[0], 1), dtype=bool)
    if err is None:
        err = np.zeros((prevPts.shape[0], 1), dtype=float)

    # initialize multiprocessing pool
    pool = Threadpool(mp.cpu_count()//2)
    # pool = Threadpool(1)

    # create neighborhood vectors
    Ix_v = np.zeros(winSize[0] * winSize[1])
    Iy_v = np.zeros(winSize[0] * winSize[1])
    It_v = np.zeros(winSize[0] * winSize[1])
    Ixyt_v = (Ix_v, Iy_v, It_v)

    # create weight matrix
    W = cv.getGaussianKernel(winSize[0] * winSize[1], -1)
    W = np.diagflat(W)

    prev_pyramid = buildOpticalFlowPyramid(prevImg, maxLevel)
    next_pyramid = buildOpticalFlowPyramid(nextImg, maxLevel)

    pyr = buildPyrLvls(prev_pyramid, next_pyramid)

    # run all point calculations in parallel
    result = pool.starmap(calcNextPt, [
        (pyr, Ixyt_v, W,
         nextPt, st, error, winSize,
         criteria, err_type, minEigThreshold)
        for nextPt, st, error in zip(nextPts, status, err)])

    j = 0
    for nextPt, st, error in result:
        nextPts[j] = nextPt
        status[j] = st
        err[j] = error
        # print(nextPt, st, error)
        j += 1

    # close up pool
    pool.close()
    pool.join()

    return nextPts, status, err


def buildPyrLvls(prev_pyramid, next_pyramid):
    pyr = []
    # iterate from highest to lowest level
    for i in range(len(prev_pyramid)-1, -1, -1):
        # get derivatives
        Ix = cv.Scharr(prev_pyramid[i], -1, 1, 0)
        Iy = cv.Scharr(prev_pyramid[i], -1, 0, 1)
        It = np.subtract(next_pyramid[i], prev_pyramid[i])
        pyr.append((Ix, Iy, It))
    return pyr


def calcNextPt(pyr, Ixyt_v, W,
               nextPt, st, error, winSize,
               criteria, err_type, minEigThreshold):
    if not st:
        return nextPt, st, error

    it_crit = True
    loc_error = error
    loc_st = st
    loc_Pt = np.rint(nextPt * 1/(2**len(pyr)))

    for i in range(len(pyr)-1, -1, -1):
        loc_Pt = np.rint(loc_Pt * 2)
        Ixyt = pyr[i]
        k = -1
        delta_uv = 0

        while(it_crit):
            prev_loc_Pt = np.copy(loc_Pt)
            prev_loc_st = np.copy(loc_st)
            prev_loc_error = np.copy(loc_error)

            loc_Pt, loc_st, loc_error = calcNeighborhoodFlow(
                                            Ixyt, Ixyt_v, W,
                                            prev_loc_Pt, loc_st, winSize,
                                            err_type, minEigThreshold)

            # in case the new flow vector doesn't work out
            if not loc_st:
                # print(prev_loc_Pt, prev_loc_st, prev_loc_error)
                loc_Pt = prev_loc_Pt
                loc_st = prev_loc_st
                loc_error = prev_loc_error
                break

            k += 1

            # iteration break check
            if criteria[0] == cv.TERM_CRITERIA_COUNT:
                it_crit = k < criteria[1]
            elif criteria[0] == cv.TERM_CRITERIA_EPS:
                uv = loc_Pt - prev_loc_Pt
                delta_uv = np.sqrt(uv[0][0]*uv[0][0] + uv[0][1]*uv[0][1])
                it_crit = delta_uv > criteria[1]
            elif criteria[0] == cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT:
                uv = loc_Pt - prev_loc_Pt
                delta_uv = np.sqrt(uv[0][0]*uv[0][0] + uv[0][1]*uv[0][1])
                it_crit = delta_uv > criteria[2] and k < criteria[1]

    return loc_Pt, loc_st, loc_error


def calcNeighborhoodFlow(Ixyt, Ixyt_v, W,
                         prevPt, st, winSize,
                         err_type, minEigThreshold):
    Pt = np.zeros((1, 2))
    error = np.zeros((1, 1), dtype=float)

    # in case the previous point already failed
    if not st:
        return Pt, st, error

    st = np.zeros((1, 1), dtype=bool)

    Ix, Iy, It = Ixyt
    Ix_v, Iy_v, It_v = Ixyt_v

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
    except ValueError:
        # print("Failed to copy neighborhood points!")
        return Pt, st, error

    Ixyt_v = (Ix_v, Iy_v, It_v)

    Pt, st, error = calcFlowVector(Ixyt_v, prevPt, W,
                                   winSize,
                                   err_type, minEigThreshold)
    return Pt, st, error


def calcFlowVector(Ixyt_v, prevPt, W,
                   winSize,
                   err_type, minEigThreshold=None):
    Ix_v, Iy_v, It_v = Ixyt_v

    Pt = np.zeros((1, 2))
    error = np.zeros(1, dtype=float)
    st = np.zeros(1, dtype=bool)

    # computes with weighted least squares method S_T.W.y=S_T.W.S.p
    y = np.array(-It_v)
    S = np.concatenate([Ix_v, Iy_v], axis=1)
    num_win_elem = winSize[0] * winSize[1]

    if (err_type == cv.OPTFLOW_LK_GET_MIN_EIGENVALS
            or minEigThreshold is not None):
        try:
            eig = np.linalg.eigvals(S.T.dot(S))

            min_eig = np.amin(eig) / num_win_elem

            if minEigThreshold is not None and min_eig < minEigThreshold:
                return Pt, st, error

        except np.linalg.LinAlgError:
            return Pt, st, error

    try:
        # x = np.linalg.lstsq(S.T.dot(W).dot(S), S.T.dot(W).dot(y), rcond=-1)
        x = np.linalg.lstsq(S, y, rcond=-1)
        uv = x[0].reshape((1, 2))
    except np.linalg.LinAlgError:
        return Pt, st, error

    # filter out points that are undetectably far away
    if np.any(uv > winSize[0]//2):
        return Pt, st, error

    if err_type == cv.OPTFLOW_LK_GET_MIN_EIGENVALS:
        error[0] = min_eig
    else:
        error[0] = np.sum(uv) / num_win_elem

    Pt[0][0] = np.rint(prevPt[0][0] + uv[0][0])
    Pt[0][1] = np.rint(prevPt[0][1] + uv[0][1])

    return Pt, np.ones(1, dtype=bool), error


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
