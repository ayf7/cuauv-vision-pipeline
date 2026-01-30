#!/usr/bin/env python3
import cv2
import numpy as np

from vision.core.base import ModuleBase
from vision.utils.feature import outer_contours
from vision.utils.transform import rect_kernel, morph_remove_noise


class BinDetector(ModuleBase):
    def process(self, direction, img):
        # 1. Threshold for beige background (HSV range)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_beige = np.array([10, 20, 60])
        upper_beige = np.array([30, 100, 255])
        mask = cv2.inRange(hsv, lower_beige, upper_beige)

        # 4. Visualize the mask (overlay)
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlayed = cv2.addWeighted(img, 0.7, mask_vis, 0.3, 0)

        # 2. Denoise with morphology
        kernel = rect_kernel(5)
        cleaned = morph_remove_noise(mask, kernel)

        # 3. Get outer contours
        contours = outer_contours(cleaned)

        best_rect = None
        best_score = float("inf")

        # for contour in contours:
        #     rect = cv2.minAreaRect(contour)
        #     (center, (w, h), angle) = rect

        #     if w * h < 500:  # Ignore very small areas
        #         continue

        #     aspect_ratio = max(w, h) / min(w, h)
        #     if aspect_ratio < 1.0 or aspect_ratio > 3.0:
        #         continue

        #     # Score based only on how close to 2:1
        #     ratio_score = abs(aspect_ratio - 2.0)
        #     if ratio_score < best_score:
        #         best_score = ratio_score
        #         best_rect = rect

        # # 7. Draw best rectangle (if any)
        # if best_rect is not None:
        #     box_points = cv2.boxPoints(best_rect)
        #     box_points = np.int0(box_points)
        #     cv2.drawContours(overlayed, [box_points], 0, (0, 255, 0), 4)

        #     ((cx, cy), (w, h), theta) = best_rect
        #     print(f"[BinDetector] Box center: ({cx:.1f}, {cy:.1f}), angle: {theta:.1f}, w={w:.1f}, h={h:.1f}")

        # 7. Store all valid rectangles that meet criteria
        valid_rects = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            (center, (w, h), angle) = rect

            if w * h < 500:  # Skip tiny blobs
                continue

            aspect_ratio = max(w, h) / min(w, h)
            if 1.0 <= aspect_ratio <= 3.0:  # Accept 2:1ish boxes
                valid_rects.append(rect)

        # 8. Draw all valid rectangles
        for rect in valid_rects:
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            cv2.drawContours(overlayed, [box_points], 0, (0, 255, 0), 4)

            ((cx, cy), (w, h), theta) = rect
            # print(f"[BinDetector] Rect center: ({cx:.1f}, {cy:.1f}), angle: {theta:.1f}, w={w:.1f}, h={h:.1f}")

        # 8. Post overlayed image showing threshold + detection
        self.post("bins", overlayed)


if __name__ == "__main__":
    BinDetector(video_sources=["forward"], tuners=[])()
