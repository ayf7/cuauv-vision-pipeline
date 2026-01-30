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


class SlalomOBB(HandlerBase):

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
        pole_red_results: List[OBBData],
        pole_white_results: List[OBBData],
    ):
        # print(pole_red_results, pole_white_results)

        slalom_group = shm.yolo_slalom
        slalom_group_results = slalom_group.get()

        # Draw all detected red poles
        for red_pole in pole_red_results:
            if red_pole.confidence >= self.tuners["slalom_threshold"]:
                points = [
                    (red_pole.x1, red_pole.y1),
                    (red_pole.x2, red_pole.y2),
                    (red_pole.x3, red_pole.y3),
                    (red_pole.x4, red_pole.y4),
                ]
                tl, tr, bl, br = order_points(points)
                draw_polylines(
                    img, [br, tr, tl, bl], color=Color.RED(), isClosed=True, thickness=2
                )

        # Draw all detected white poles
        for white_pole in pole_white_results:
            if white_pole.confidence >= self.tuners["slalom_threshold"]:
                points = [
                    (white_pole.x1, white_pole.y1),
                    (white_pole.x2, white_pole.y2),
                    (white_pole.x3, white_pole.y3),
                    (white_pole.x4, white_pole.y4),
                ]
                tl, tr, bl, br = order_points(points)
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.WHITE(),
                    isClosed=True,
                    thickness=2,
                )

        # ── Red pole mapping ──
        red_pole = None
        red_center_x = 0.0
        red_center_y = 0.0

        if not pole_red_results:
            slalom_group_results.slalom_red_visible = 0
        else:
            # Choose the red pole with the bottom-most coordinates (highest y-value)
            def get_bottom_y(pole):
                points = [
                    (pole.x1, pole.y1),
                    (pole.x2, pole.y2),
                    (pole.x3, pole.y3),
                    (pole.x4, pole.y4),
                ]
                return max(point[1] for point in points)

            red_pole = max(pole_red_results, key=get_bottom_y)

            if red_pole.confidence < self.tuners["slalom_threshold"]:
                slalom_group_results.slalom_red_visible = 0
            else:
                points = [
                    (red_pole.x1, red_pole.y1),
                    (red_pole.x2, red_pole.y2),
                    (red_pole.x3, red_pole.y3),
                    (red_pole.x4, red_pole.y4),
                ]
                tl, tr, bl, br = order_points(points)

                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.RED(),
                    isClosed=True,
                    thickness=12,
                )

                # Normalize coordinates and add to SHM
                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                slalom_group_results.slalom_red_visible = 1
                slalom_group_results.slalom_red_confidence = red_pole.confidence
                slalom_group_results.slalom_red_bottom_right_y = br[0]
                slalom_group_results.slalom_red_bottom_right_x = br[1]
                slalom_group_results.slalom_red_top_right_y = tr[0]
                slalom_group_results.slalom_red_top_right_x = tr[1]
                slalom_group_results.slalom_red_top_left_y = tl[0]
                slalom_group_results.slalom_red_top_left_x = tl[1]
                slalom_group_results.slalom_red_bottom_left_y = bl[0]
                slalom_group_results.slalom_red_bottom_left_x = bl[1]
                slalom_group_results.slalom_red_center_y = (
                    br[0] + bl[0] + tr[0] + tl[0]
                ) / 4
                slalom_group_results.slalom_red_center_x = (
                    br[1] + bl[1] + tr[1] + tl[1]
                ) / 4
                slalom_group_results.slalom_red_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

                red_center_x = slalom_group_results.slalom_red_center_x
                red_center_y = slalom_group_results.slalom_red_center_y

        # ── White poles mapping (left and right relative to red) ──
        if (
            not pole_white_results
            or not pole_red_results
            or red_pole is None
            or red_pole.confidence < self.tuners["slalom_threshold"]
        ):
            slalom_group_results.slalom_left_visible = 0
            slalom_group_results.slalom_right_visible = 0
        else:
            # Filter white poles by confidence
            valid_white_poles = [
                pole
                for pole in pole_white_results
                if pole.confidence >= self.tuners["slalom_threshold"]
            ]

            # Get red pole center x coordinate for comparison
            red_points = [
                (red_pole.x1, red_pole.y1),
                (red_pole.x2, red_pole.y2),
                (red_pole.x3, red_pole.y3),
                (red_pole.x4, red_pole.y4),
            ]
            red_center_x_pixel = sum(point[0] for point in red_points) / 4

            # Separate poles into left and right of red pole
            left_poles = []
            right_poles = []

            for pole in valid_white_poles:
                points = [
                    (pole.x1, pole.y1),
                    (pole.x2, pole.y2),
                    (pole.x3, pole.y3),
                    (pole.x4, pole.y4),
                ]
                center_x = sum(point[0] for point in points) / 4

                if center_x < red_center_x_pixel:  # Left of red pole
                    left_poles.append(pole)
                elif center_x > red_center_x_pixel:  # Right of red pole
                    right_poles.append(pole)

            # Select bottom-most pole from each side
            def get_bottom_y(pole):
                points = [
                    (pole.x1, pole.y1),
                    (pole.x2, pole.y2),
                    (pole.x3, pole.y3),
                    (pole.x4, pole.y4),
                ]
                return max(point[1] for point in points)

            left_pole = max(left_poles, key=get_bottom_y) if left_poles else None
            right_pole = max(right_poles, key=get_bottom_y) if right_poles else None

            # Process left pole
            if left_pole is not None:
                points = [
                    (left_pole.x1, left_pole.y1),
                    (left_pole.x2, left_pole.y2),
                    (left_pole.x3, left_pole.y3),
                    (left_pole.x4, left_pole.y4),
                ]
                tl, tr, bl, br = order_points(points)
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.YELLOW(),
                    isClosed=True,
                    thickness=12,
                )

                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                slalom_group_results.slalom_left_visible = 1
                slalom_group_results.slalom_left_confidence = left_pole.confidence
                slalom_group_results.slalom_left_bottom_right_y = br[0]
                slalom_group_results.slalom_left_bottom_right_x = br[1]
                slalom_group_results.slalom_left_top_right_y = tr[0]
                slalom_group_results.slalom_left_top_right_x = tr[1]
                slalom_group_results.slalom_left_top_left_y = tl[0]
                slalom_group_results.slalom_left_top_left_x = tl[1]
                slalom_group_results.slalom_left_bottom_left_y = bl[0]
                slalom_group_results.slalom_left_bottom_left_x = bl[1]
                slalom_group_results.slalom_left_center_y = (
                    br[0] + bl[0] + tr[0] + tl[0]
                ) / 4
                slalom_group_results.slalom_left_center_x = (
                    br[1] + bl[1] + tr[1] + tl[1]
                ) / 4
                slalom_group_results.slalom_left_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )
            else:
                slalom_group_results.slalom_left_visible = 0

            # Process right pole
            if right_pole is not None:
                points = [
                    (right_pole.x1, right_pole.y1),
                    (right_pole.x2, right_pole.y2),
                    (right_pole.x3, right_pole.y3),
                    (right_pole.x4, right_pole.y4),
                ]
                tl, tr, bl, br = order_points(points)
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.CYAN(),
                    isClosed=True,
                    thickness=12,
                )

                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                slalom_group_results.slalom_right_visible = 1
                slalom_group_results.slalom_right_confidence = right_pole.confidence
                slalom_group_results.slalom_right_bottom_right_y = br[0]
                slalom_group_results.slalom_right_bottom_right_x = br[1]
                slalom_group_results.slalom_right_top_right_y = tr[0]
                slalom_group_results.slalom_right_top_right_x = tr[1]
                slalom_group_results.slalom_right_top_left_y = tl[0]
                slalom_group_results.slalom_right_top_left_x = tl[1]
                slalom_group_results.slalom_right_bottom_left_y = bl[0]
                slalom_group_results.slalom_right_bottom_left_x = bl[1]
                slalom_group_results.slalom_right_center_y = (
                    br[0] + bl[0] + tr[0] + tl[0]
                ) / 4
                slalom_group_results.slalom_right_center_x = (
                    br[1] + bl[1] + tr[1] + tl[1]
                ) / 4
                slalom_group_results.slalom_right_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )
            else:
                slalom_group_results.slalom_right_visible = 0

        # Write everything back
        slalom_group.set(slalom_group_results)
        self.post("slalom handler", img)

    def post_grayscale(self, img: np.ndarray):
        gray_img, _ = bgr_to_gray(img)
        self.post("slalom handler", gray_img)
