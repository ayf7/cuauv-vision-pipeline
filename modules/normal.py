#!/usr/bin/env python3
from vision.core.base import ModuleBase
from vision.core import tuners
from vision.utils.transform import decode_normal
from vision.utils.draw import draw_circle
from typing import Tuple
import cv2
import math

module_options = [
    tuners.IntTuner('x', 400, 0, 720),
    tuners.IntTuner('y', 400, 0, 1280),
]


class Normal(ModuleBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = self.y = self.z = 0
        self.count = 0

    def reset(self):
        self.x = self.y = self.z = self.count = 0

    def process(self, *img):
        img_transformed = decode_normal(img[0])

        coord_x: int = self.tuners['x']
        coord_y: int = self.tuners['y']

        x, y, z = img_transformed[coord_x, coord_y]

        # Keep a moving average of x, y, z
        self.count += 1
        self.x = ((self.x * (self.count - 1)) + x) / self.count
        self.y = ((self.y * (self.count - 1)) + y) / self.count
        self.z = ((self.z * (self.count - 1)) + z) / self.count
        print(self.x, self.y, self.z)
        print(img[0][coord_x, coord_y])

        draw_circle(img[0], (coord_y, coord_x), 10, thickness=10)
        self.post("point", img[0])


if __name__ == '__main__':
    Normal("normal", module_options)()
