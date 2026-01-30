#!/usr/bin/env python3
import numpy as np
from vision.core.base import ModuleBase, sources
from vision.core.tuners import IntTuner
from vision.utils.color import bgr_to_lab, range_threshold
from vision.utils.feature import outer_contours, contour_centroid, contour_area
from vision.utils.draw import draw_contours
from vision.utils.transform import morph_close_holes, morph_remove_noise, rect_kernel
import shm

module_tuners = [
    IntTuner("thresh_min", 0, 0, 255),
    IntTuner("thresh_max", 255, 0, 255)
]

class BuoyLAB(ModuleBase):

    @sources("zed[forward]", "zed[normal]")
    def process_img(self, image, normal):
        # LAB A channel
        lab, (lab_l, lab_a, lab_b) = bgr_to_lab(image)

        # COLOR THRESHOLDING
        threshed = range_threshold(
            lab_a,
            self.tuners["thresh_min"],
            self.tuners["thresh_max"]
        )
        self.post("threshed", threshed, "GRAY")

        # some cleanup cleanup
        kernel = rect_kernel(5)
        cleaned = morph_remove_noise(threshed, kernel)
        cleaned = morph_close_holes(cleaned, kernel)
        self.post("threshed_cleaned", cleaned, "GRAY")

        # CONTOUR DETECTION AND PROCESSING
        contours = outer_contours(threshed)
        draw_contours(image, contours, thickness=10)
        contour = self.extract_most_likely_contour() # logic omitted

        # extract information
        x, y = contour_centroid(contour)
        area = contour_area(contour)
        ny, nx = self.normalize((y, x))

        # WRITING TO SHM
        shm.red_buoy_results.center_x.set(nx)
        shm.red_buoy_results.center_x.set(ny)
        shm.red_buoy_results.area.set(area)

        self.post("contours", image)


if __name__ == "__main__":
    BuoyLAB(["zed"], module_tuners)()
