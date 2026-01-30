#!/usr/bin/env python3
import cv2
import numpy as np

import shm
import shm.camera_calibration
from vision.core.base import ModuleBase
from conf.vehicle import cameras, is_mainsub
from vision.utils.draw import draw_rect, draw_text
from vision.core.tuners import IntTuner, BoolTuner, DoubleTuner
from vision.utils.color import bgr_to_hsv, bgr_to_lab, lab_to_bgr, bgr_to_gray

def _safe_get_int(name, default):
    try:
        return getattr(shm.camera_calibration, name).get()
    except Exception:
        return default
    
    
def get_module_options(direction: str):
    brightness = {
        (True, True): 152,
        (True, False): 172,
        (False, True): 180,
        (False, False): 150,
    }[(is_mainsub, direction == "forward")]

    try:
        width = getattr(cameras, direction).width
        height = getattr(cameras, direction).height
    except AttributeError:
        width = 640
        height = 512

    return [
        BoolTuner(f"debug", False),
        BoolTuner(f"enable", True),
        DoubleTuner(f"adjustment_smoothing", 0.1, 0.1, 10),
        IntTuner(f"target_brightness", 30, 0, 300),
        IntTuner(f"bright_acceptable_error", 20, 0, 255),
        IntTuner("target_r", 100, 0, 255),
        IntTuner("target_b", 100, 0, 255),
        IntTuner("white_balance_acceptable_error", 30, 0, 100),
        DoubleTuner("zed_white_balance", 4850.0, 2800.0, 6500.0),

        # NEW: manual-only image controls (defaults pulled from SHM when possible)
        IntTuner("zed_contrast",    _safe_get_int("zed_contrast",    4), 0, 8),
        IntTuner("zed_hue",         _safe_get_int("zed_hue",         0), 0, 11),
        IntTuner("zed_saturation",  _safe_get_int("zed_saturation",  8), 0, 8),
        IntTuner("zed_gamma",       _safe_get_int("zed_gamma",       4), 0, 8),
        IntTuner("zed_sharpness",   _safe_get_int("zed_sharpness",   4), 0, 8),
    ]


class AutoCalibrateZed(ModuleBase):
    def __init__(self, direction):
        super().__init__(direction, get_module_options(direction[0]))
        self.direction = direction
        self.directions = [direction]
        self.module_name = f"AutoCalibrateZed_{direction[0]}"
        self.reset_debug_text()

    def get_shm(self, group, name):
        return getattr(group, f"{name}").get()

    def set_shm(self, group, name, value):
        return getattr(group, f"{name}").set(value)

    def update_value(self, img, shm_var, multiplier, min_val, max_val):
        current_val = self.get_shm(shm.camera_calibration, shm_var)
        target = current_val * multiplier
        target = max(min_val, target)
        target = min(max_val, target)

        r = np.exp(-0.1 / self.tuners["adjustment_smoothing"])
        target += r * (current_val - target)
        target = round(target)

        self.set_shm(shm.camera_calibration, shm_var, target)

    def update_whitebalance(self, img, shm_var, k=1000):
        avg_b, avg_g, avg_r = np.mean(img, axis=(0, 1))
        current_val = self.get_shm(shm.camera_calibration, shm_var)

        gain_r = self.tuners["target_r"] / max(avg_r, 1e-6)
        gain_b = self.tuners["target_b"] / max(avg_b, 1e-6)

        diff = gain_r - gain_b
        r = np.exp(-0.01 * (abs(diff)))
        delta = r * k * (diff)
        if abs(delta) < 50:
            delta = 0

        self.set_shm(
            shm.camera_calibration,
            shm_var,
            int(np.clip(current_val + delta, 2800, 6500)),
        )

    def reset_debug_text(self):
        self.debug_text_pos = (25, 40)

    def draw_debug_text(self, img, text):
        if self.tuners["debug"]:
            average_color = np.mean(img[self.debug_text_pos[1] + 16, :, :])
            if average_color < 127:
                text_color = (255, 255, 255)  # white
                text_outline = (0, 0, 0)  # black
            else:
                text_color = (0, 0, 0)  # black
                text_outline = (255, 255, 255)  # white

            draw_text(
                img, text, self.debug_text_pos, 0.8, color=text_outline, thickness=8
            )
            draw_text(
                img, text, self.debug_text_pos, 0.8, color=text_color, thickness=2
            )

    def process(self, direction, img):
        self.post(direction, img)
        self.reset_debug_text()
        enabled = self.tuners["enable"]

        try:
            shm.camera_calibration.zed_contrast.set(   int(self.tuners["zed_contrast"]))
            shm.camera_calibration.zed_hue.set(        int(self.tuners["zed_hue"]))
            shm.camera_calibration.zed_saturation.set( int(self.tuners["zed_saturation"]))
            shm.camera_calibration.zed_gamma.set(      int(self.tuners["zed_gamma"]))
            shm.camera_calibration.zed_sharpness.set(  int(self.tuners["zed_sharpness"]))
        except Exception:
            pass

        _, lab_img = bgr_to_lab(img)
        (lab_l, _, _) = lab_img

        if enabled:
            # Currently not autotuning Gamma and Sharpness

            # Exposure
            brightness_average = np.average(lab_l)
            target_bright = self.tuners["target_brightness"]
            self.draw_debug_text(
                img, f"  1) brightness average: {brightness_average:.2f}"
            )
            if (
                abs(brightness_average - target_bright)
                > self.tuners["bright_acceptable_error"]
            ):
                self.update_value(
                    img, "zed_exposure", target_bright / brightness_average, 1, 100
                )

            # Gain
            target_bright = self.tuners["target_brightness"]
            self.draw_debug_text(
                img, f"  1) brightness average: {brightness_average:.2f}"
            )
            if (
                abs(brightness_average - target_bright)
                > self.tuners["bright_acceptable_error"]
            ):
                self.update_value(
                    img, "zed_gain", target_bright / brightness_average, 0, 100
                )

            # Brightness
            target_bright = self.tuners["target_brightness"]
            self.draw_debug_text(
                img, f"  1) brightness average: {brightness_average:.2f}"
            )
            if (
                abs(brightness_average - target_bright)
                > self.tuners["bright_acceptable_error"]
            ):
                self.update_value(
                    img, "zed_brightness", target_bright / brightness_average, 0, 8
                )

            # White Balance
            getattr(shm.camera_calibration, "zed_white_balance").set(
                int(self.tuners["zed_white_balance"])
            )
            # self.update_whitebalance(img, "zed_white_balance")


# import sys
if __name__ == "__main__":
    AutoCalibrateZed(["forward"])()
