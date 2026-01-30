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


class TorpedoesOBB(HandlerBase):

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
        board_results: List[OBBData],
        shark_hole_results: List[OBBData],
        saw_hole_results: List[OBBData],
    ):

        board_group = shm.yolo_torpedoes_board
        board_group_results = board_group.get()

        if not board_results:
            # Board not detected - set board visibility to 0
            board_group_results.board_visible = 0
        else:
            # Heuristic: choose the most confidence.
            torpedo = max(board_results, key=lambda x: x.confidence)

            if torpedo.confidence < self.tuners["torpedo_threshold"]:
                # Board confidence too low - set board visibility to 0
                board_group_results.board_visible = 0
            else:
                # Board detected with sufficient confidence - process it
                # Extract coordinates of the board
                points = [
                    (torpedo.x1, torpedo.y1),
                    (torpedo.x2, torpedo.y2),
                    (torpedo.x3, torpedo.y3),
                    (torpedo.x4, torpedo.y4),
                ]
                tl, tr, bl, br = order_points(points)

                # Draw
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.LIME(),
                    isClosed=True,
                    thickness=3,
                )

                # Normalize coordinates and add to SHM
                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                board_group_results.board_visible = 1
                board_group_results.board_confidence = torpedo.confidence
                board_group_results.board_bottom_right_y = br[0]
                board_group_results.board_bottom_right_x = br[1]
                board_group_results.board_top_right_y = tr[0]
                board_group_results.board_top_right_x = tr[1]
                board_group_results.board_top_left_y = tl[0]
                board_group_results.board_top_left_x = tl[1]
                board_group_results.board_bottom_left_y = bl[0]
                board_group_results.board_bottom_left_x = bl[1]
                board_group_results.board_center_y = (br[0] + bl[0] + tr[0] + tl[0]) / 4
                board_group_results.board_center_x = (br[1] + bl[1] + tr[1] + tl[1]) / 4

                shm.relay.point_x.set(((tl[1] + bl[1]) / 2 + (tr[1] + br[1]) / 2) / 2)
                shm.relay.point_y.set(((tl[0] + tr[0]) / 2 + (bl[0] + br[0]) / 2) / 2)
                board_group_results.board_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # ── Shark‐hole mapping ──
        if not shark_hole_results:
            board_group_results.shark_visible = 0
        else:
            shark_hole = max(shark_hole_results, key=lambda x: x.confidence)

            # Check confidence threshold (using same threshold as board)
            if shark_hole.confidence < self.tuners["torpedo_threshold"]:
                board_group_results.shark_visible = 0
            else:
                pts = [
                    (shark_hole.x1, shark_hole.y1),
                    (shark_hole.x2, shark_hole.y2),
                    (shark_hole.x3, shark_hole.y3),
                    (shark_hole.x4, shark_hole.y4),
                ]
                tl, tr, bl, br = order_points(pts)

                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.BLUE(),
                    isClosed=True,
                    thickness=3,
                )

                # normalize & reverse into [y, x]
                br, tr, tl, bl = [self.normalize(reverse(p)) for p in (br, tr, tl, bl)]

                board_group_results.shark_visible = 1
                board_group_results.shark_confidence = shark_hole.confidence
                board_group_results.shark_top_left_x = tl[1]
                board_group_results.shark_top_left_y = tl[0]
                board_group_results.shark_top_right_x = tr[1]
                board_group_results.shark_top_right_y = tr[0]
                board_group_results.shark_bottom_left_x = bl[1]
                board_group_results.shark_bottom_left_y = bl[0]
                board_group_results.shark_bottom_right_x = br[1]
                board_group_results.shark_bottom_right_y = br[0]
                board_group_results.shark_center_x = (tl[1] + tr[1] + bl[1] + br[1]) / 4
                board_group_results.shark_center_y = (tl[0] + tr[0] + bl[0] + br[0]) / 4
                board_group_results.shark_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # ── Saw‐hole mapping ──
        if not saw_hole_results:
            board_group_results.saw_visible = 0
        else:
            saw_hole = max(saw_hole_results, key=lambda x: x.confidence)

            # Check confidence threshold (using same threshold as board)
            if saw_hole.confidence < self.tuners["torpedo_threshold"]:
                board_group_results.saw_visible = 0
            else:
                pts = [
                    (saw_hole.x1, saw_hole.y1),
                    (saw_hole.x2, saw_hole.y2),
                    (saw_hole.x3, saw_hole.y3),
                    (saw_hole.x4, saw_hole.y4),
                ]
                tl, tr, bl, br = order_points(pts)

                draw_polylines(
                    img, [br, tr, tl, bl], color=Color.RED(), isClosed=True, thickness=3
                )

                br, tr, tl, bl = [self.normalize(reverse(p)) for p in (br, tr, tl, bl)]

                board_group_results.saw_visible = 1
                board_group_results.saw_confidence = saw_hole.confidence
                board_group_results.saw_top_left_x = tl[1]
                board_group_results.saw_top_left_y = tl[0]
                board_group_results.saw_top_right_x = tr[1]
                board_group_results.saw_top_right_y = tr[0]
                board_group_results.saw_bottom_left_x = bl[1]
                board_group_results.saw_bottom_left_y = bl[0]
                board_group_results.saw_bottom_right_x = br[1]
                board_group_results.saw_bottom_right_y = br[0]
                board_group_results.saw_center_x = (tl[1] + tr[1] + bl[1] + br[1]) / 4
                board_group_results.saw_center_y = (tl[0] + tr[0] + bl[0] + br[0]) / 4
                board_group_results.saw_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # write everything back
        board_group.set(board_group_results)
        self.post("torpedoes handler", img)

    def post_grayscale(self, img: np.ndarray):
        gray_img, _ = bgr_to_gray(img)
        self.post("torpedoes handler", gray_img)
