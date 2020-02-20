#!/usr/bin/env python

import numpy as np
import cv2
import os
import argparse

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame0", "-f0", required=True, type=str, default="middlebury/Urban/frame10.png")
    parser.add_argument("--frame1", "-f1", required=True, type=str, default="middlebury/Urban/frame11.png")

    args = parser.parse_args()

    # print(args.frame0)
    # print(args.frame1)
    frame0 = cv2.imread(args.frame0, cv2.IMREAD_GRAYSCALE)
    frame1 = cv2.imread(args.frame1, cv2.IMREAD_GRAYSCALE)

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    p0 = cv2.goodFeaturesToTrack(frame0, mask = None, **feature_params)
    mask = np.zeros_like(frame0)

    p1, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame0, (a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame', img)
    cv2.waitKey(0)
