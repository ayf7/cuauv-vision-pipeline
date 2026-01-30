import math
from typing import List, Tuple

import numpy as np

import shm
from vision.yolo.data import OBBData
from vision.utils.color import bgr_to_gray
from vision.yolo.utils import order_points
from vision.core.handlers import HandlerBase
from vision.utils.draw import Color, draw_polylines


def reverse(tup):
    return tup[1], tup[0]


class GateBehindOBB(HandlerBase):
    def compute_area_normalized(self, corners, img_shape):
        """Compute area of quadrilateral from normalized coordinates using shoelace formula.

        The area is scaled so that the full image area equals 1.0.
        Since normalization uses width as denominator for both x and y coordinates:
        - x_range = 1.0 (from -0.5 to 0.5)
        - y_range = height/width (from -height/(2*width) to height/(2*width))
        - scaling_factor = width/height to normalize area to 1.0 for full image

        Args:
            corners: List of (y, x) coordinate tuples in normalized space
            img_shape: (height, width, channels) from image.shape
        """
        x = [pt[1] for pt in corners]
        y = [pt[0] for pt in corners]
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
        source_name: str,
        img: np.ndarray,
        gate_behind_results: List[OBBData],
    ):
        """
        Args:
            source_name: identifier for the frame source (kept for parity with other handlers)
            img: BGR image
            gate_behind_results: list of OBBData for the Roboflow class 'gate_behind'
        """
        gb_group = shm.yolo_gate_behind
        gb_data = gb_group.get()

        # Count
        gb_data.num_gate_behind_detected = len(gate_behind_results)

        # Visibility / geometry
        if not gate_behind_results:
            gb_data.gate_behind_visible = 0
            gb_data.gate_behind_confidence = 0
        else:
            best = max(gate_behind_results, key=lambda x: x.confidence)
            if best.confidence < self.tuners["bins_threshold"]:
                gb_data.gate_behind_visible = 0
            else:
                points = [
                    (best.x1, best.y1),
                    (best.x2, best.y2),
                    (best.x3, best.y3),
                    (best.x4, best.y4),
                ]
                tl, tr, bl, br = order_points(points)

                # Draw oriented box for visualization
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.CYAN(),
                    isClosed=True,
                    thickness=3,
                )

                # Normalize into (y, x) order like other handlers
                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                gb_data.gate_behind_visible = 1
                gb_data.gate_behind_confidence = best.confidence

                # Corner coords
                gb_data.gate_behind_bottom_right_y = br[0]
                gb_data.gate_behind_bottom_right_x = br[1]
                gb_data.gate_behind_top_right_y = tr[0]
                gb_data.gate_behind_top_right_x = tr[1]
                gb_data.gate_behind_top_left_y = tl[0]
                gb_data.gate_behind_top_left_x = tl[1]
                gb_data.gate_behind_bottom_left_y = bl[0]
                gb_data.gate_behind_bottom_left_x = bl[1]

                # Center + area
                gb_data.gate_behind_center_y = (br[0] + bl[0] + tr[0] + tl[0]) / 4
                gb_data.gate_behind_center_x = (br[1] + bl[1] + tr[1] + tl[1]) / 4
                gb_data.gate_behind_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # Write back + publish
        gb_group.set(gb_data)
        self.post("gate_behind handler", img)

    def post_grayscale(self, img: np.ndarray):
        gray_img, _ = bgr_to_gray(img)
        self.post("gate_behind handler", gray_img)
