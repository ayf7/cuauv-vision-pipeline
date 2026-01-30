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


class BinsOBB(HandlerBase):

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
        bin_shark_results: List[OBBData],
        bin_saw_results: List[OBBData],
        bin_results: List[OBBData],
    ):

        bins_group = shm.yolo_bins
        bins_group_results = bins_group.get()

        # ── Shark side mapping ──
        if not bin_shark_results:
            bins_group_results.bin_shark_visible = 0
            bins_group_results.bin_shark_confidence = 0
        else:
            # Pick the most confident bin_shark
            bin_shark = max(bin_shark_results, key=lambda x: x.confidence)

            if bin_shark.confidence < self.tuners["bins_threshold"]:
                bins_group_results.bin_shark_visible = 0
            else:
                points = [
                    (bin_shark.x1, bin_shark.y1),
                    (bin_shark.x2, bin_shark.y2),
                    (bin_shark.x3, bin_shark.y3),
                    (bin_shark.x4, bin_shark.y4),
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

                bins_group_results.bin_shark_visible = 1
                bins_group_results.bin_shark_confidence = bin_shark.confidence
                bins_group_results.bin_shark_bottom_right_y = br[0]
                bins_group_results.bin_shark_bottom_right_x = br[1]
                bins_group_results.bin_shark_top_right_y = tr[0]
                bins_group_results.bin_shark_top_right_x = tr[1]
                bins_group_results.bin_shark_top_left_y = tl[0]
                bins_group_results.bin_shark_top_left_x = tl[1]
                bins_group_results.bin_shark_bottom_left_y = bl[0]
                bins_group_results.bin_shark_bottom_left_x = bl[1]
                bins_group_results.bin_shark_center_y = (
                    br[0] + bl[0] + tr[0] + tl[0]
                ) / 4
                bins_group_results.bin_shark_center_x = (
                    br[1] + bl[1] + tr[1] + tl[1]
                ) / 4
                bins_group_results.bin_shark_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # ── Saw side mapping ──
        if not bin_saw_results:
            bins_group_results.bin_saw_visible = 0
            bins_group_results.bin_saw_confidence = 0
        else:
            # Pick the most confident bin_saw
            bin_saw = max(bin_saw_results, key=lambda x: x.confidence)

            if bin_saw.confidence < self.tuners["bins_threshold"]:
                bins_group_results.bin_saw_visible = 0
            else:
                points = [
                    (bin_saw.x1, bin_saw.y1),
                    (bin_saw.x2, bin_saw.y2),
                    (bin_saw.x3, bin_saw.y3),
                    (bin_saw.x4, bin_saw.y4),
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

                bins_group_results.bin_saw_visible = 1
                bins_group_results.bin_saw_confidence = bin_saw.confidence
                bins_group_results.bin_saw_bottom_right_y = br[0]
                bins_group_results.bin_saw_bottom_right_x = br[1]
                bins_group_results.bin_saw_top_right_y = tr[0]
                bins_group_results.bin_saw_top_right_x = tr[1]
                bins_group_results.bin_saw_top_left_y = tl[0]
                bins_group_results.bin_saw_top_left_x = tl[1]
                bins_group_results.bin_saw_bottom_left_y = bl[0]
                bins_group_results.bin_saw_bottom_left_x = bl[1]
                bins_group_results.bin_saw_center_y = (
                    br[0] + bl[0] + tr[0] + tl[0]
                ) / 4
                bins_group_results.bin_saw_center_x = (
                    br[1] + bl[1] + tr[1] + tl[1]
                ) / 4
                bins_group_results.bin_saw_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # ── Bin mapping ──
        if not bin_results:
            bins_group_results.bin_visible = 0
            bins_group_results.bin_confidence = 0
        else:
            # Pick the most confident bin
            bin_obj = max(bin_results, key=lambda x: x.confidence)

            if bin_obj.confidence < self.tuners["bins_threshold"]:
                bins_group_results.bin_visible = 0
            else:
                points = [
                    (bin_obj.x1, bin_obj.y1),
                    (bin_obj.x2, bin_obj.y2),
                    (bin_obj.x3, bin_obj.y3),
                    (bin_obj.x4, bin_obj.y4),
                ]
                tl, tr, bl, br = order_points(points)

                # Draw
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.GREEN(),
                    isClosed=True,
                    thickness=3,
                )

                # Normalize coordinates and add to SHM
                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                bins_group_results.bin_visible = 1
                bins_group_results.bin_confidence = bin_obj.confidence
                bins_group_results.bin_bottom_right_y = br[0]
                bins_group_results.bin_bottom_right_x = br[1]
                bins_group_results.bin_top_right_y = tr[0]
                bins_group_results.bin_top_right_x = tr[1]
                bins_group_results.bin_top_left_y = tl[0]
                bins_group_results.bin_top_left_x = tl[1]
                bins_group_results.bin_bottom_left_y = bl[0]
                bins_group_results.bin_bottom_left_x = bl[1]
                bins_group_results.bin_center_y = (br[0] + bl[0] + tr[0] + tl[0]) / 4
                bins_group_results.bin_center_x = (br[1] + bl[1] + tr[1] + tl[1]) / 4
                bins_group_results.bin_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # write everything back
        print(bins_group_results.bin_shark_area)
        bins_group.set(bins_group_results)
        self.post("bins handler", img)

    def post_grayscale(self, img: np.ndarray):
        gray_img, _ = bgr_to_gray(img)
        self.post("bins handler", gray_img)
