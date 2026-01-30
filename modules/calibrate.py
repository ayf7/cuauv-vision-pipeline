#!/usr/bin/env python3
import time
import ctypes

import cv2
import numpy as np

import shm
from vision.core import tuners
from conf.vehicle import cameras, is_mainsub
from vision.core.base import ModuleBase, VideoSource, sources

directions = list(cameras.keys())
print(directions)

opts = []

DEFAULT_DOUBLE_MAX = 100.0
DEFAULT_DOUBLE_MIN = 0.0
DEFAULT_INT_MAX = 50
DEFAULT_INT_MIN = 0


def build_opts():
    # FIX: since orion is still technically minisub yet has DVL.
    if True:
        return [
            tuners.DoubleTuner(
                "downward_blue_gain", -1, DEFAULT_DOUBLE_MIN, DEFAULT_DOUBLE_MAX
            ),
            tuners.DoubleTuner(
                "downward_exposure", -1, DEFAULT_DOUBLE_MIN, DEFAULT_DOUBLE_MAX
            ),
            tuners.DoubleTuner(
                "downward_green_gain", -1, DEFAULT_DOUBLE_MIN, DEFAULT_DOUBLE_MAX
            ),
            tuners.DoubleTuner(
                "downward_red_gain", -1, DEFAULT_DOUBLE_MIN, DEFAULT_DOUBLE_MAX
            ),
            tuners.DoubleTuner(
                "forward_blue_gain", -1, DEFAULT_DOUBLE_MIN, DEFAULT_DOUBLE_MAX
            ),
            tuners.DoubleTuner(
                "forward_exposure", -1, DEFAULT_DOUBLE_MIN, DEFAULT_DOUBLE_MAX
            ),
            tuners.DoubleTuner(
                "forward_green_gain", -1, DEFAULT_DOUBLE_MIN, DEFAULT_DOUBLE_MAX
            ),
            tuners.DoubleTuner(
                "forward_red_gain", -1, DEFAULT_DOUBLE_MIN, DEFAULT_DOUBLE_MAX
            ),
            tuners.IntTuner("zed_brightness", 6, 0, 8),
            tuners.IntTuner("zed_contrast", 4, 0, 8),
            tuners.IntTuner("zed_hue", 0, 0, 11),
            tuners.IntTuner("zed_saturation", 2, 0, 8),
            tuners.IntTuner("zed_gamma", 4, 0, 8),
            tuners.IntTuner("zed_sharpness", 4, 0, 8),
            tuners.IntTuner("zed_white_balance", 5000, 3500, 6500),
            tuners.IntTuner("zed_exposure", 29, 0, 100),
            tuners.IntTuner("zed_gain", 73, 0, 100),
        ]

    else:
        for o, t in shm.camera_calibration._fields:
            print(o)
            if t == ctypes.c_double:
                opts.append(
                    tuners.DoubleTuner(
                        o,
                        getattr(shm.camera_calibration, o).get(),
                        DEFAULT_DOUBLE_MIN,
                        DEFAULT_DOUBLE_MAX,
                    )
                )
            elif t == ctypes.c_int:
                opts.append(
                    tuners.IntTuner(
                        o,
                        getattr(shm.camera_calibration, o).get(),
                        DEFAULT_INT_MIN,
                        DEFAULT_INT_MAX,
                    )
                )
        return opts


class Calibrate(ModuleBase):
    def __init__(self, directions):
        super().__init__(directions, build_opts())
        self.prev = {}

    def _apply_tuners(self):
        for o, t in shm.camera_calibration._fields:
            opt_val = self.tuners[o]
            if o not in self.prev or opt_val != self.prev[o]:
                getattr(shm.camera_calibration, o).set(opt_val)
                self.prev[o] = opt_val

    @sources("zed[forward2]", "zed[forward]")
    def show_zed_images(self, img_fwd2, img_fwd):
        self._apply_tuners()
        self.post("forward2", img_fwd2)
        self.post("forward", img_fwd)

    @sources("zed[depth]")
    def show_depth(self, depth):
        self._apply_tuners()
        depth_u8 = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        self.post("depth", depth_u8)

    @sources("zed[normal]")
    def show_normal(self, normal):
        self._apply_tuners()
        normal_u8 = np.clip((normal * 255), 0, 255.0).astype(np.uint8)
        self.post("normal", normal_u8)

    # Fallback for any uncovered sources
    def process(self, name, mat):
        self._apply_tuners()
        self.post(name, mat)


if __name__ == "__main__":
    # With plane names carried by CMF, only the base source is needed.
    Calibrate(["zed"])()
