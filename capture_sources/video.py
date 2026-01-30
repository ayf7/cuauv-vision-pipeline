#!/usr/bin/env python3
import os
import cv2
import argparse
from typing import Tuple, List
from vision.core.capture_source import CaptureSource, FpsLimiter


def video_to_directions(fps_limiter: FpsLimiter, args: Tuple[str, List[str], bool]):
    source = args[0]
    directions = args[1]
    loop = args[2]

    cap = cv2.VideoCapture(source)  # type: ignore
    target_fps = cap.get(cv2.CAP_PROP_FPS)  # type: ignore


    for curr_time in fps_limiter.rate(target_fps):
        _, next_img = cap.read()

        if next_img is None:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # type: ignore
                continue
            else:
                break

        for direction in directions:
            yield direction, curr_time, next_img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        f"{__file__}", description='CLI to pipe video frames into a vision module')
    parser.add_argument('--loop', action='store_true', help='loop video forever')
    parser.add_argument('sources', nargs="+", type=str,
                        help="specify video sources and their directions in the format 'filepath:dir1,dir2'")
    args = parser.parse_args()

    targets: List[Tuple[str, str]] = [
        tuple(item.split(':')) for item in args.sources
    ]

    # check if files are valid
    for file, _ in targets:
        if not os.path.exists(file):
            print(f"filepath '{file}' is not valid")
            exit(1)

    cs = CaptureSource()
    for file, directions in targets:
        lst = directions.split(',')
        cs.register_capture_udl(
            ' '.join(lst), video_to_directions, args=(file, lst, args.loop))
    cs.run_event_loop()
