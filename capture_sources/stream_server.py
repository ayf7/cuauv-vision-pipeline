#!/usr/bin/env python3
import cv2
import time
import argparse
from typing import Tuple
from nanomsg import Socket, PUB  # type: ignore

from vision.core.bindings.camera_message_framework import BlockAccessor, SUCCESS


SERVER_ADDR = "tcp://0.0.0.0:8081"


def pack_image(img):
    _, jpeg = cv2.imencode(  # type: ignore
        '.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 100))  # type: ignore
    return jpeg.tobytes()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Server to stream images from a camera to clients')
    parser.add_argument('direction', metavar='dir', type=str,
                        help='camera direction (e.g., forward or downward)')
    args = parser.parse_args()

    with BlockAccessor(args.direction) as a, Socket(PUB) as sock:
        while True:
            _, data, status = a.read_frame()
            if status == SUCCESS:
                sock.send(pack_image(data))
                time.sleep(0.1)
            else:
                time.sleep(0.05)

