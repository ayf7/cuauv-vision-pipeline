import math
from typing import List

import numpy as np

import shm
from vision.yolo.data import OBBData
from vision.utils.color import bgr_to_gray
from vision.yolo.utils import order_points
from vision.core.handlers import HandlerBase
from vision.utils.draw import Color, draw_polylines


def reverse(tup):
    return tup[1], tup[0]


class ManipulatorOBB(HandlerBase):
    def compute_area_normalized(self, corners, img_shape):
        """Compute area of quadrilateral from normalized coordinates using shoelace formula."""
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
        source_name: str,
        img: np.ndarray,
        spoon_results: List[OBBData],
        cup_results: List[OBBData],
        pink_basket_results: List[OBBData],
        yellow_basket_results: List[OBBData],
    ):
        manip_group = shm.yolo_manipulator
        manip_data = manip_group.get()

        # --- Spoon ---
        manip_data.num_spoons_detected = len(spoon_results)
        if not spoon_results:
            manip_data.spoon_visible = 0
            manip_data.spoon_confidence = 0
        else:
            best_spoon = max(spoon_results, key=lambda x: x.confidence)
            if best_spoon.confidence < self.tuners["manipulator_threshold"]:
                manip_data.spoon_visible = 0
            else:
                points = [
                    (best_spoon.x1, best_spoon.y1),
                    (best_spoon.x2, best_spoon.y2),
                    (best_spoon.x3, best_spoon.y3),
                    (best_spoon.x4, best_spoon.y4),
                ]
                tl, tr, bl, br = order_points(points)
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.DEEPPINK(),
                    isClosed=True,
                    thickness=3,
                )

                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                manip_data.spoon_visible = 1
                manip_data.spoon_confidence = best_spoon.confidence
                manip_data.spoon_top_left_x = tl[1]
                manip_data.spoon_top_left_y = tl[0]
                manip_data.spoon_top_right_x = tr[1]
                manip_data.spoon_top_right_y = tr[0]
                manip_data.spoon_bottom_left_x = bl[1]
                manip_data.spoon_bottom_left_y = bl[0]
                manip_data.spoon_bottom_right_x = br[1]
                manip_data.spoon_bottom_right_y = br[0]
                manip_data.spoon_center_x = (br[1] + bl[1] + tr[1] + tl[1]) / 4
                manip_data.spoon_center_y = (br[0] + bl[0] + tr[0] + tl[0]) / 4
                manip_data.spoon_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # --- Cup ---
        manip_data.num_cups_detected = len(cup_results)
        if not cup_results:
            manip_data.cup_visible = 0
            manip_data.cup_confidence = 0
        else:
            best_cup = max(cup_results, key=lambda x: x.confidence)
            if best_cup.confidence < self.tuners["manipulator_threshold"]:
                manip_data.cup_visible = 0
            else:
                points = [
                    (best_cup.x1, best_cup.y1),
                    (best_cup.x2, best_cup.y2),
                    (best_cup.x3, best_cup.y3),
                    (best_cup.x4, best_cup.y4),
                ]
                tl, tr, bl, br = order_points(points)
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.YELLOW(),
                    isClosed=True,
                    thickness=3,
                )

                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                manip_data.cup_visible = 1
                manip_data.cup_confidence = best_cup.confidence
                manip_data.cup_top_left_x = tl[1]
                manip_data.cup_top_left_y = tl[0]
                manip_data.cup_top_right_x = tr[1]
                manip_data.cup_top_right_y = tr[0]
                manip_data.cup_bottom_left_x = bl[1]
                manip_data.cup_bottom_left_y = bl[0]
                manip_data.cup_bottom_right_x = br[1]
                manip_data.cup_bottom_right_y = br[0]
                manip_data.cup_center_x = (br[1] + bl[1] + tr[1] + tl[1]) / 4
                manip_data.cup_center_y = (br[0] + bl[0] + tr[0] + tl[0]) / 4
                manip_data.cup_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # --- Pink Basket ---
        if not pink_basket_results:
            manip_data.pink_basket_visible = 0
            manip_data.pink_basket_confidence = 0
        else:
            best_pink = max(pink_basket_results, key=lambda x: x.confidence)
            if best_pink.confidence < self.tuners["manipulator_threshold"]:
                manip_data.pink_basket_visible = 0
            else:
                points = [
                    (best_pink.x1, best_pink.y1),
                    (best_pink.x2, best_pink.y2),
                    (best_pink.x3, best_pink.y3),
                    (best_pink.x4, best_pink.y4),
                ]
                tl, tr, bl, br = order_points(points)
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.HOTPINK(),
                    isClosed=True,
                    thickness=3,
                )

                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                manip_data.pink_basket_visible = 1
                manip_data.pink_basket_confidence = best_pink.confidence
                manip_data.pink_basket_top_left_x = tl[1]
                manip_data.pink_basket_top_left_y = tl[0]
                manip_data.pink_basket_top_right_x = tr[1]
                manip_data.pink_basket_top_right_y = tr[0]
                manip_data.pink_basket_bottom_left_x = bl[1]
                manip_data.pink_basket_bottom_left_y = bl[0]
                manip_data.pink_basket_bottom_right_x = br[1]
                manip_data.pink_basket_bottom_right_y = br[0]
                manip_data.pink_basket_center_x = (br[1] + bl[1] + tr[1] + tl[1]) / 4
                manip_data.pink_basket_center_y = (br[0] + bl[0] + tr[0] + tl[0]) / 4
                manip_data.pink_basket_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        # --- Yellow Basket ---
        if not yellow_basket_results:
            manip_data.yellow_basket_visible = 0
            manip_data.yellow_basket_confidence = 0
        else:
            best_yellow = max(yellow_basket_results, key=lambda x: x.confidence)
            if best_yellow.confidence < self.tuners["manipulator_threshold"]:
                manip_data.yellow_basket_visible = 0
            else:
                points = [
                    (best_yellow.x1, best_yellow.y1),
                    (best_yellow.x2, best_yellow.y2),
                    (best_yellow.x3, best_yellow.y3),
                    (best_yellow.x4, best_yellow.y4),
                ]
                tl, tr, bl, br = order_points(points)
                draw_polylines(
                    img,
                    [br, tr, tl, bl],
                    color=Color.LEMON(),
                    isClosed=True,
                    thickness=3,
                )

                br = self.normalize(reverse(br))
                tr = self.normalize(reverse(tr))
                tl = self.normalize(reverse(tl))
                bl = self.normalize(reverse(bl))

                manip_data.yellow_basket_visible = 1
                manip_data.yellow_basket_confidence = best_yellow.confidence
                manip_data.yellow_basket_top_left_x = tl[1]
                manip_data.yellow_basket_top_left_y = tl[0]
                manip_data.yellow_basket_top_right_x = tr[1]
                manip_data.yellow_basket_top_right_y = tr[0]
                manip_data.yellow_basket_bottom_left_x = bl[1]
                manip_data.yellow_basket_bottom_left_y = bl[0]
                manip_data.yellow_basket_bottom_right_x = br[1]
                manip_data.yellow_basket_bottom_right_y = br[0]
                manip_data.yellow_basket_center_x = (br[1] + bl[1] + tr[1] + tl[1]) / 4
                manip_data.yellow_basket_center_y = (br[0] + bl[0] + tr[0] + tl[0]) / 4
                manip_data.yellow_basket_area = self.compute_area_normalized(
                    [br, tr, tl, bl], img.shape
                )

        manip_group.set(manip_data)
        self.post("manipulator handler", img)
