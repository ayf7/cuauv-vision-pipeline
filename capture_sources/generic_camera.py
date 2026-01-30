#!/usr/bin/env python3
import cv2
import argparse
from vision.core.capture_source import CaptureSource, FpsLimiter

CAMERA_DIRECTION = "forward"
CAMERA_INDEX = 0
CAMERA_FPS = 15


def generic_capture(fps_limiter: FpsLimiter, _):
    camera = cv2.VideoCapture(CAMERA_INDEX)  # type: ignore

    for acquisition_time in fps_limiter.rate(CAMERA_FPS):
        _, image = camera.read()
        if image == None:
            return

        yield CAMERA_DIRECTION, acquisition_time, image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        f"{__file__}", description='CLI to pipe generic hardware device into vision module')

    parser.add_argument('direction', default='forward',
                        nargs='?', type=str, help='direction to write frames to (default=forward)')
    parser.add_argument('--device', default=0, type=int,
                        help='Device index (default=0)')
    parser.add_argument('--fps', default=15, type=int,
                        help='Capture speed (default=15)')
    args = parser.parse_args()

    CAMERA_DIRECTION = args.direction
    CAMERA_INDEX = args.device
    CAMERA_FPS = args.fps

    cs = CaptureSource()
    cs.register_capture_udl(CAMERA_DIRECTION, generic_capture)
    cs.run_event_loop()
