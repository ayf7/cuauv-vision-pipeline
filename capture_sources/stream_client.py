#!/usr/bin/env python3
import time
import cv2
import argparse
import numpy as np
from typing import Tuple
from nanomsg import Socket, SUB, SUB_SUBSCRIBE # type: ignore

from vision.core.capture_source import CaptureSource, FpsLimiter


def unpack_image(msg):
    buffer = np.fromstring(msg, dtype='uint8') # type: ignore
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) # type: ignore

def stream_udl(limiter: FpsLimiter, args: Tuple[str, str]):
    address = args[0]
    direction = args[1] + '_stream'


    server_addr = 'tcp://{}:8081'.format(address)
    sock = Socket(SUB)
    sock.connect(server_addr)
    sock.set_string_option(SUB, SUB_SUBSCRIBE, '')

    # socket blocks so just don't have fps limiter
    for _ in limiter.rate(0):
        img = unpack_image(sock.recv())
        yield direction, int(time.monotonic() * 1000), img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Client that streams images from a camera to emulate a local capture source')
    parser.add_argument('direction', metavar='dir', type=str,
                        help='camera direction (e.g., forward or downward)')
    parser.add_argument('server_addr', type=str,
                        help='address of the streaming server')
    args = parser.parse_args()

    cs = CaptureSource()
    cs.register_capture_udl(args.direction, stream_udl, (args.server_addr, args.direction))
    cs.run_event_loop()
