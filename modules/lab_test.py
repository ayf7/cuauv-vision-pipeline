#!/usr/bin/env python3
import numpy as np

import shm
from vision.core.base import ModuleBase
from vision.utils.color import bgr_to_lab

try:
    from vision.capture_sources.zed import ZED_MAX_DISTANCE, ZED_MIN_DISTANCE
except ImportError:
    ZED_MIN_DISTANCE = 0
    ZED_MAX_DISTANCE = 1


class LAB(ModuleBase):
    def process(self, direction, image):

        if direction == "depth":
            image -= ZED_MIN_DISTANCE
            image /= ZED_MAX_DISTANCE - ZED_MIN_DISTANCE
            image *= 255
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = np.repeat(image, 3, axis=2)
        elif direction == "normal":
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        self.post(direction, image)

        lab, lab_split = bgr_to_lab(image)
        lab_l, lab_a, lab_b = lab_split

        self.post("LAB", lab, "LAB")
        self.post("LAB L", lab_l, "LAB")
        self.post("LAB A", lab_a, "LAB")
        self.post("LAB B", lab_b, "LAB")


if __name__ == "__main__":
    LAB(["depth"])()
