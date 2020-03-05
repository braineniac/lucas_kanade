#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import multiprocessing as mp
from multiprocessing.dummy import Pool as Threadpool

from lucas_kanade import calcOpticalFlowPyrLK, buildPyrLvls, calcNextPt, \
 calcNeighborhoodFlow, buildOpticalFlowPyramid, pyrDown
import lucas_kanade


def calcFlowVector(Ixyt_v, prevPt, W,
                   winSize,
                   err_type, minEigThreshold=None):
    uv = np.zeros((2, 1))
    A = np.zeros((2, 2))

    nextPt = np.zeros((1, 2))
    error = np.zeros(1, dtype=float)
    st = np.zeros(1, dtype=bool)

    Ix_v, Iy_v, It_v = Ixyt_v
    Pt_x, Pt_y = prevPt[0]

    num_win_elem = winSize[0] * winSize[1]

    del_I = np.concatenate([Ix_v.T,
                            Iy_v.T,
                            Pt_x * Ix_v.T,
                            Pt_y * Ix_v.T,
                            Pt_x * Iy_v.T,
                            Pt_y * Iy_v.T])
    G = del_I.dot(del_I.T)
    b = del_I.dot(It_v)

    if (err_type == cv.OPTFLOW_LK_GET_MIN_EIGENVALS
            or minEigThreshold is not None):
        try:
            eig = np.linalg.eigvals(G.T.dot(G))

            min_eig = np.amin(eig) / num_win_elem

            if minEigThreshold is not None and min_eig < minEigThreshold:
                return nextPt, st, error

        except np.linalg.LinAlgError:
            return nextPt, st, error

    try:
        x = np.linalg.lstsq(G, b, rcond=-1)
        x = x[0]

        uv[0] = x[0]
        uv[0] = x[1]
        A[0][0] = 1 + x[2]
        A[0][1] = x[3]
        A[1][0] = x[4]
        A[1][1] = 1 + x[5]

    except np.linalg.LinAlgError:
        return nextPt, st, error

    if err_type == cv.OPTFLOW_LK_GET_MIN_EIGENVALS:
        error[0] = min_eig
    else:
        error[0] = np.sum(uv) / num_win_elem

    nextPt = np.rint(A.dot(prevPt.T) + uv)
    nextPt = nextPt.T

    return nextPt, np.ones(1, dtype=bool), error


# overwrite function
lucas_kanade.calcFlowVector = calcFlowVector
