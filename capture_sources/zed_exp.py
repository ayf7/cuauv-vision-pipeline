#!/usr/bin/env python3
import os
import time
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

ZED_CAMERA_FPS: int = 15
ZED_IMAGE_FPS: int = 15
ZED_DEPTH_FPS: int = 15
ZED_NORMAL_FPS: int = 15


def image_udl(fps_limiter: FpsLimiter, args: Tuple[sl.Camera]):
    def to_rgb(x) -> np.ndarray:
        return cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)  # type: ignore

    zed = args[0]
    left_mat = sl.Mat()
    right_mat = sl.Mat()

    run_params = sl.RuntimeParameters(
        enable_fill_mode=True, remove_saturated_areas=False
    )

    for acquisition_time in fps_limiter.rate(ZED_IMAGE_FPS):

        if zed.grab(run_params) != sl.ERROR_CODE.SUCCESS:
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

        # depth_ocv = np.where(np.isnan(depth_ocv), 0, depth_ocv)

        yield ZED_DEPTH_DIRECTION, acquisition_time, depth_ocv


def normal_udl(fps_limiter: FpsLimiter, args: Tuple[sl.Camera]):
    zed = args[0]

    normal_mat = sl.Mat()
    stage_keys = [
        ("retrieve", "retrieve_measure"),
        ("get_data", "get_data_contig"),
        ("scale", "scale"),
        ("nan_to_num", "nan_to_num"),
    ]
    stage_totals = {key: 0.0 for key, _ in stage_keys}
    total_time_sum = 0.0
    iteration = 0
    last_frame_time = None
    fps_sum = 0.0
    fps_samples = 0

    for acquisition_time in fps_limiter.rate(ZED_NORMAL_FPS):
        iteration += 1

        loop_start = time.perf_counter()

        stage_start = time.perf_counter()
        zed.retrieve_measure(normal_mat, sl.MEASURE.NORMALS)
        retrieve_time = time.perf_counter()

        normal_map_start = time.perf_counter()
        normal_map = normal_mat.get_data()  # [..., :3]
        normal_map = np.ascontiguousarray(normal_map)
        get_data_time = time.perf_counter()

        # normal_map = (normal_map + 1) / 2.0  # Range from [-1, 1] to [0, 1]
        normal_map += 1
        normal_map /= 2.0
        scale_time = time.perf_counter()

        # normal_map = np.where(np.isnan(normal_map), 0, normal_map)

        nan_to_num_time = time.perf_counter()

        stage_durations = {
            "retrieve": retrieve_time - stage_start,
            "get_data": get_data_time - normal_map_start,
            "scale": scale_time - get_data_time,
            "nan_to_num": nan_to_num_time - scale_time,
        }
        loop_duration = nan_to_num_time - loop_start

        if loop_duration > 0:
            total_time_sum += loop_duration
            for key in stage_totals:
                stage_totals[key] += stage_durations[key]

            current_percentages = {
                key: (stage_durations[key] / loop_duration) * 100.0
                for key in stage_totals
            }
            running_percentages = {
                key: (stage_totals[key] / total_time_sum) * 100.0
                for key in stage_totals
            }
        else:
            current_percentages = {key: 0.0 for key in stage_totals}
            running_percentages = {key: 0.0 for key in stage_totals}

        now = time.monotonic()
        if last_frame_time is not None:
            frame_delta = now - last_frame_time
            if frame_delta > 0:
                current_fps = 1.0 / frame_delta
                fps_sum += current_fps
                fps_samples += 1
                fps_display = f"{current_fps:.2f}"
            else:
                fps_display = "inf"
        else:
            fps_display = "n/a"
        last_frame_time = now

        loop_duration_ms = loop_duration * 1000.0
        avg_fps_display = f"{fps_sum / fps_samples:.2f}" if fps_samples else "n/a"

        breakdown = "; ".join(
            f"{label}: {current_percentages[key]:5.1f}% (avg {running_percentages[key]:5.1f}%)"
            for key, label in stage_keys
        )
        print(
            f"Normal timing #{iteration}: total {loop_duration_ms:.3f} ms, "
            f"FPS {fps_display} (avg {avg_fps_display}) | {breakdown}"
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
