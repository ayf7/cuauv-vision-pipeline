#!/usr/bin/env python3

import os
import cv2
import argparse
import itertools

from typing import Tuple

from vision.core.capture_source import CaptureSource, FpsLimiter


def image_direction_capture(fps: FpsLimiter, t: Tuple[argparse.Namespace]):
    args = t[0]

    images = []
    for name in os.listdir(args.directory):
        filename = os.path.join(args.directory, name)
        image = cv2.imread(filename)  # type: ignore
        if image is not None:
            images.append(image)

    if args.no_loop:
        # use a regular iterator if we shouldn't loop
        image_iter = iter(images)
    else:
        # use a looping iterator if we should loop
        image_iter = itertools.cycle(images)

    for acquisition_time in fps.rate(args.fps):
        next_image = next(image_iter, None)

        if next_image is None:
            return
        
        yield args.direction, acquisition_time, next_image


def main():
    parser = argparse.ArgumentParser(
        description='Process a directory of images as a video')
    parser.add_argument('direction', help='direction to act as a camera')
    parser.add_argument('directory', help='directory of images')
    parser.add_argument('--fps', default=60.0, type=float,
                        help='FPS to run the module at')
    parser.add_argument('--no-loop', action='store_true',
                        help='disable looping')

    args = parser.parse_args()

    cs = CaptureSource()
    cs.register_capture_udl(
        args.direction, image_direction_capture, args=(args, ))
    cs.run_event_loop()


if __name__ == '__main__':
    main()
