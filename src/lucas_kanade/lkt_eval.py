#!/usr/bin/env python3

import os
import argparse
import cv2 as cv
import numpy as np

import lucas_kanade as lk


class LKTEvaluator:
    def __init__(self, eval_name="", use_opencv=False, corner_params=[], lk_params=[]):
        self.frames = self.load_frames(eval_name)
        self.corner_params = corner_params
        self.lk_params = lk_params
        self.flow = []
        self.use_opencv = use_opencv

    def load_frames(self, eval_name=None):
        module_path = os.path.dirname(os.path.realpath(__file__))
        relative_path = "/../../middlebury/" + eval_name + "/"
        full_path = module_path + relative_path

        frame0 = cv.imread(full_path + "frame10.png", cv.IMREAD_GRAYSCALE)
        frame1 = cv.imread(full_path + "frame11.png", cv.IMREAD_GRAYSCALE)

        if frame0 is None or frame1 is None:
            print("Data folder not found!")
            exit(-1)

        return (frame0, frame1)

    def evaluate(self):
        # run Shi Tomasi corner detection for finding good points to track
        p0 = cv.goodFeaturesToTrack(self.frames[0], mask=None, **self.corner_params)

        # run Lucas-Kanade-Method
        if self.use_opencv:
            p1, st, err = cv.calcOpticalFlowPyrLK(self.frames[0], self.frames[1], p0, None, **self.lk_params)
        else:
            p1, st, err = lk.calcOpticalFlowPyrLK(self.frames[0], self.frames[1], p0, None, **self.lk_params)

        # print(p1[st == 1])
        # reorder good points as (start, end)
        for start, end in zip(p0[st == 1], p1[st == 1]):
            self.flow.append((tuple(start), tuple(end)))

    def plot_flow(self):
        mask = np.zeros_like(self.frames[0])
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        for line in self.flow:
            start, end = line
            mask = cv.arrowedLine(mask, start, end, (0, 0, 255), 1)
        frame0 = cv.cvtColor(self.frames[0], cv.COLOR_GRAY2BGR)
        plot = cv.add(frame0, mask)
        cv.imshow('Optical flow', plot)
        cv.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", type=str, default="Dimetrodon")
    parser.add_argument("--opencv", action="store_true")
    args = parser.parse_args()

    # Parameters
    corner_params = dict(maxCorners=100,
                         qualityLevel=0.3,
                         minDistance=7,
                         blockSize=7
                         )
    lk_params = dict(winSize=(5, 5),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    lkt_evaluator = LKTEvaluator(args.name, args.opencv, corner_params, lk_params)
    lkt_evaluator.evaluate()
    lkt_evaluator.plot_flow()
