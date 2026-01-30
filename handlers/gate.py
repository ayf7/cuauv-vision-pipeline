import math
from typing import List, Tuple

import numpy as np

import shm
from vision.yolo.data import OBBData
from vision.utils.color import bgr_to_gray
from vision.yolo.utils import order_points
from vision.core.handlers import HandlerBase
from vision.utils.draw import Color, draw_circle, draw_polylines


def reverse(tup):
    return tup[1], tup[0]


def intify(tup):
    return int(tup[0]), int(tup[1])


class GateOBB(HandlerBase):

    def compute_area_normalized(self, corners, img_shape):
        """Compute area of quadrilateral from normalized coordinates using shoelace formula

        The area is scaled so that the full image area equals 1.0.
        Since normalization uses width as denominator for both x and y coordinates:
        - x_range = 1.0 (from -0.5 to 0.5)
        - y_range = height/width (from -height/(2*width) to height/(2*width))
        - scaling_factor = width/height to normalize area to 1.0 for full image

        Args:
            corners: List of (y, x) coordinate tuples in normalized space
            img_shape: (height, width) tuple from image.shape
        """
        x = [corner[1] for corner in corners]  # x coordinates
        y = [corner[0] for corner in corners]  # y coordinates
        area = 0.0
        n = len(x)
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j]
            area -= x[j] * y[i]

        raw_area = abs(area) / 2.0
        height, width, _ = img_shape
        scaling_factor = width / height
        return raw_area * scaling_factor

    def process(
        self,
        direction: str,
        img: np.ndarray,
        shark_results: List[OBBData],
        saw_results: List[OBBData],
    ):

        gate_group = shm.yolo_gate
        gate_group_results = gate_group.get()

        # ── Shark mapping ──
        if not shark_results:
            gate_group_results.shark_visible = 0
            gate_group_results.shark_confidence = 0
        else:
            # Pick the most confident shark
            shark = max(shark_results, key=lambda x: x.confidence)

            if shark.confidence < self.tuners["gate_threshold"]:
                gate_group_results.shark_visible = 0
            else:
                points = [
                    (shark.x1, shark.y1),
                    (shark.x2, shark.y2),
                    (shark.x3, shark.y3),
                    (shark.x4, shark.y4),
                ]
                tl, tr, bl, br = order_points(points)

                # Draw
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.BLUE(),
                    isClosed=True,
                    thickness=3,
                )

                # Normalize coordinates and add to SHM
                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                gate_group_results.shark_visible = 1
                gate_group_results.shark_confidence = shark.confidence
                gate_group_results.shark_bottom_right_y = br[0]
                gate_group_results.shark_bottom_right_x = br[1]
                gate_group_results.shark_top_right_y = tr[0]
                gate_group_results.shark_top_right_x = tr[1]
                gate_group_results.shark_top_left_y = tl[0]
                gate_group_results.shark_top_left_x = tl[1]
                gate_group_results.shark_bottom_left_y = bl[0]
                gate_group_results.shark_bottom_left_x = bl[1]
                gate_group_results.shark_center_y = (br[0] + bl[0] + tr[0] + tl[0]) / 4
                gate_group_results.shark_center_x = (br[1] + bl[1] + tr[1] + tl[1]) / 4
                gate_group_results.shark_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # ── Saw mapping ──
        if not saw_results:
            gate_group_results.saw_visible = 0
        else:
            # Pick the most confident saw
            saw = max(saw_results, key=lambda x: x.confidence)

            if saw.confidence < self.tuners["gate_threshold"]:
                gate_group_results.saw_visible = 0
            else:
                points = [
                    (saw.x1, saw.y1),
                    (saw.x2, saw.y2),
                    (saw.x3, saw.y3),
                    (saw.x4, saw.y4),
                ]
                tl, tr, bl, br = order_points(points)

                # Draw
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.RED(),
                    isClosed=True,
                    thickness=3,
                )

                # Normalize coordinates and add to SHM
                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                gate_group_results.saw_visible = 1
                gate_group_results.saw_confidence = saw.confidence
                gate_group_results.saw_bottom_right_y = br[0]
                gate_group_results.saw_bottom_right_x = br[1]
                gate_group_results.saw_top_right_y = tr[0]
                gate_group_results.saw_top_right_x = tr[1]
                gate_group_results.saw_top_left_y = tl[0]
                gate_group_results.saw_top_left_x = tl[1]
                gate_group_results.saw_bottom_left_y = bl[0]
                gate_group_results.saw_bottom_left_x = bl[1]
                gate_group_results.saw_center_y = (br[0] + bl[0] + tr[0] + tl[0]) / 4
                gate_group_results.saw_center_x = (br[1] + bl[1] + tr[1] + tl[1]) / 4
                gate_group_results.saw_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # write everything back
        gate_group.set(gate_group_results)
        self.post("gate handler", img)

    def post_grayscale(self, img: np.ndarray):
        gray_img, _ = bgr_to_gray(img)
        self.post("gate handler", gray_img)
