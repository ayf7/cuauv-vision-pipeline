#!/usr/bin/env python3
import os
import time
from typing import Dict, Tuple

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
ZED_IMAGE_FPS: int = 15
ZED_DEPTH_FPS: int = 15
ZED_NORMAL_FPS: int = 15

# Benchmark reporting interval (seconds), roughly matches prior behavior (~3s)
BENCH_REPORT_INTERVAL_S: float = 3.0


def to_rgb(x) -> np.ndarray:
    return cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)  # type: ignore


class StageBenchmark:
    def __init__(self):
        self.ema: Dict[str, float] = {}
        self.last_time = None
        self.alpha = 0.1
        self.iter = 0
        self.fps_ema = 0.0

    def update_fps(self):
        now = time.perf_counter()
        if self.last_time:
            inst_fps = 1 / (now - self.last_time)
            self.fps_ema = (
                (1 - self.alpha) * self.fps_ema + self.alpha * inst_fps
                if self.iter > 1
                else inst_fps
            )
        self.last_time = now
        self.iter += 1
        return self.fps_ema

    def update_stage(self, name: str, duration: float):
        prev = self.ema.get(name, duration)
        self.ema[name] = (1 - self.alpha) * prev + self.alpha * duration

    def summary(self) -> str:
        stages = " | ".join(f"{k}:{v*1000:.1f}ms" for k, v in self.ema.items())
        return f"FPS: {self.fps_ema:.1f} | {stages}"


def all_udl(fps_limiter: FpsLimiter, args: Tuple[sl.Camera], benchmarking=True):
    assert ZED_IMAGE_FPS == ZED_NORMAL_FPS == ZED_DEPTH_FPS
    zed = args[0]

    mats = {
        "left": sl.Mat(),
        "right": sl.Mat(),
        "depth": sl.Mat(),
        "normal": sl.Mat(),
    }

    run_params = sl.RuntimeParameters(
        enable_fill_mode=True, remove_saturated_areas=False
    )
    benchmark = StageBenchmark() if benchmarking else None
    last_report = time.perf_counter() if benchmarking else None

    for acquisition_time in fps_limiter.rate(ZED_IMAGE_FPS):

        if zed.grab(run_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED grab error!")

        # Measure each retrieve individually, then aggregate.
        retrieve_total = 0.0

        t_s = time.perf_counter()
        zed.retrieve_image(mats["left"], sl.VIEW.LEFT)
        t_e = time.perf_counter()
        retrieve_total += t_e - t_s

        t_s = time.perf_counter()
        zed.retrieve_image(mats["right"], sl.VIEW.RIGHT)
        t_e = time.perf_counter()
        retrieve_total += t_e - t_s

        t_s = time.perf_counter()
        zed.retrieve_measure(mats["depth"], sl.MEASURE.DEPTH)
        t_e = time.perf_counter()
        retrieve_total += t_e - t_s

        t_s = time.perf_counter()
        zed.retrieve_measure(mats["normal"], sl.MEASURE.NORMALS)
        t_e = time.perf_counter()
        retrieve_total += t_e - t_s

        if benchmarking:
            benchmark.update_stage("retrieve", retrieve_total)

        # Get CPU-side arrays and convert where needed.
        t_get_start = time.perf_counter()
        left_image = to_rgb(mats["left"].get_data())
        right_image = to_rgb(mats["right"].get_data())
        depth_ocv = mats["depth"].get_data()
        normal_map = mats["normal"].get_data()  # [..., :3]
        t_get_end = time.perf_counter()

        if benchmarking:
            benchmark.update_stage("get_data", t_get_end - t_get_start)

        normal_map = np.ascontiguousarray(normal_map)  # keep uncommented for cmf

        depth_ocv = np.where(np.isnan(depth_ocv), 0, depth_ocv)
        normal_map += 1
        normal_map /= 2.0
        normal_map = np.where(np.isnan(normal_map), 0, normal_map)
        t3 = time.perf_counter()

        if benchmarking:
            benchmark.update_stage("nan_to_num", t3 - t_get_end)

        if benchmarking:
            benchmark.update_fps()

        yield ZED_IMAGE_DIRECTION_LEFT, acquisition_time, left_image
        yield ZED_IMAGE_DIRECTION_RIGHT, acquisition_time, right_image
        yield ZED_DEPTH_DIRECTION, acquisition_time, depth_ocv
        yield ZED_NORMAL_DIRECTION, acquisition_time, normal_map

        if benchmarking:
            now = time.perf_counter()
            if now - last_report >= BENCH_REPORT_INTERVAL_S:  # type: ignore[arg-type]
                print(f"[ZED] {benchmark.summary()}")
                last_report = now


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
    cs.register_capture_udl("all udl", all_udl, (zed,))
    cs.register_logical_udl(calibrate_udl, (zed,))
    cs.run_event_loop()
