#!/usr/bin/env python3

import os
import argparse
import cv2


class LKTEvaluator:
    def __init__(self, eval_name="", eval_type="", corner_params=[], lk_params=[]):
        self.frames = self.load_frames(eval_name, eval_type)
        self.corner_params = corner_params
        self.lk_params = lk_params

    def load_frames(self, eval_name=None, eval_type=None):
        if eval_type == "two_frames":
            module_path = os.path.dirname(os.path.realpath(__file__))
            relative_path = "/../../middlebury/" + eval_type + "/" + eval_name + "/"
            full_path = module_path + relative_path
            print(full_path)

            frame0 = cv2.imread(full_path + "frame10.png", cv2.IMREAD_GRAYSCALE)
            frame1 = cv2.imread(full_path + "frame11.png", cv2.IMREAD_GRAYSCALE)

            if frame0 is None or frame1 is None:
                print("Data folder not found!")
                exit(-1)
            # cv2.imshow('frame0', frame0)
            # cv2.imshow('frame1', frame1)
            # cv2.waitKey(0)
            return (frame0, frame1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", type=str, default="Dimetrodon")
    parser.add_argument("--type", "-t", type=str, default="two_frames")
    args = parser.parse_args()

    # Parameters
    corner_params = dict(maxCorners=100,
                         qualityLevel=0.3,
                         minDistance=7,
                         blockSize=7
                         )
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    lkt_evaluator = LKTEvaluator(
        args.name, args.type, corner_params, lk_params)
