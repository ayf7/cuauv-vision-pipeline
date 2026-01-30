#!/usr/bin/env python3
import os
from typing import Tuple

import cv2
import numpy as np
import pyzed.sl as sl

import shm
from vision.core.capture_source import FpsLimiter, CaptureSource

VIDEO_SETTINGS = sl.VIDEO_SETTINGS

# zed configurations
ZED_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vision/configs/zed.conf"
)

ZED_IMAGE_DIRECTION_LEFT: str = "forward2"
ZED_IMAGE_DIRECTION_RIGHT: str = "forward"

ZED_DEPTH_DIRECTION: str = "depth"
ZED_NORMAL_DIRECTION: str = "normal"
ZED_USE_LEFT_CAMERA: bool = True

ZED_MIN_DISTANCE: float = 0.5
ZED_MAX_DISTANCE: float = 10

ZED_CAMERA_FPS: int = 30
ZED_IMAGE_FPS: int = 22
ZED_DEPTH_FPS: int = 5
ZED_NORMAL_FPS: int = 5


def image_udl(fps_limiter: FpsLimiter, args: Tuple[sl.Camera]):
    def to_rgb(x) -> np.ndarray:
        return cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)  # type: ignore

    zed = args[0]
    left_mat = sl.Mat()
    right_mat = sl.Mat()

    for acquisition_time in fps_limiter.rate(ZED_IMAGE_FPS):

        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Zed grab error")

        zed.retrieve_image(left_mat, sl.VIEW.LEFT)
        zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

        left_image = left_mat.get_data()
        right_image = right_mat.get_data()

        yield ZED_IMAGE_DIRECTION_LEFT, acquisition_time, to_rgb(left_image)
        yield ZED_IMAGE_DIRECTION_RIGHT, acquisition_time, to_rgb(right_image)


def depth_udl(fps_limiter: FpsLimiter, args: Tuple[sl.Camera]):
    zed = args[0]

    depth_mat = sl.Mat()
    for acquisition_time in fps_limiter.rate(ZED_DEPTH_FPS):
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        depth_ocv = depth_mat.get_data()

        np.nan_to_num(
            depth_ocv,
            copy=False,
            nan=ZED_MAX_DISTANCE,
            posinf=ZED_MAX_DISTANCE,
            neginf=ZED_MIN_DISTANCE,
        )

        yield ZED_DEPTH_DIRECTION, acquisition_time, depth_ocv


def normal_udl(fps_limiter: FpsLimiter, args: Tuple[sl.Camera]):
    zed = args[0]

    normal_mat = sl.Mat()
    for acquisition_time in fps_limiter.rate(ZED_NORMAL_FPS):
        zed.retrieve_measure(normal_mat, sl.MEASURE.NORMALS)

        normal_map = normal_mat.get_data()[..., :3]
        normal_map = np.ascontiguousarray(normal_map)

        # normal_map = (normal_map + 1) / 2.0  # Range from [-1, 1] to [0, 1]
        normal_map += 1
        normal_map /= 2.0

        np.nan_to_num(
            normal_map,
            copy=False,
            nan=0,
        )

        yield ZED_NORMAL_DIRECTION, acquisition_time, normal_map


def calibrate_udl(fps_limiter: FpsLimiter, args: Tuple[sl.Camera]):
    zed = args[0]

    VS = VIDEO_SETTINGS
    for _ in fps_limiter.rate(2):
        c = shm.camera_calibration.get()  # type: ignore
        zed.set_camera_settings(VS.BRIGHTNESS, c.zed_brightness)
        zed.set_camera_settings(VS.CONTRAST, c.zed_contrast)
        zed.set_camera_settings(VS.HUE, c.zed_hue)
        zed.set_camera_settings(VS.SATURATION, c.zed_saturation)
        zed.set_camera_settings(VS.GAMMA, c.zed_gamma)
        zed.set_camera_settings(VS.SHARPNESS, c.zed_sharpness)
        zed.set_camera_settings(VS.WHITEBALANCE_TEMPERATURE, c.zed_white_balance)
        zed.set_camera_settings(VS.EXPOSURE, c.zed_exposure)
        zed.set_camera_settings(VS.GAIN, c.zed_gain)
        zed.set_camera_settings(VS.AEC_AGC, 0)
        zed.set_camera_settings(VS.WHITEBALANCE_AUTO, 0)


if __name__ == "__main__":
    zed_inits_params = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        optional_settings_path=ZED_CONFIG_PATH,
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
        depth_minimum_distance=ZED_MIN_DISTANCE,
        depth_maximum_distance=ZED_MAX_DISTANCE,
        camera_resolution=sl.RESOLUTION.HD720,
        camera_fps=ZED_CAMERA_FPS,
    )
    zed = sl.Camera()

    status = zed.open(zed_inits_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    print("ZED Camera initialized. Starting frame capture...")

    cs = CaptureSource()
    cs.register_capture_udl("image udl", image_udl, (zed,))
    cs.register_capture_udl("depth udl", depth_udl, (zed,))
    cs.register_capture_udl("normal udl", normal_udl, (zed,))
    cs.register_logical_udl(calibrate_udl, (zed,))
    cs.run_event_loop()
