#!/usr/bin/env python3
import os
import time
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyzed.sl as sl

import shm
from vision.core.capture_source import FpsLimiter, CaptureSource

"""
NOTE:
A pipelined processing variant, implemented solely in Python, that enables
concurrent execution of retrieving frames and processing frames (numpy ops).
While speedup is limited due to limitations of "multithreading" in Python(<=13)
it still provides a noticeable speedup with this implementation.
"""

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

ZED_MIN_DISTANCE: float = 0
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

    def _alloc_mats():
        return {
            "left": sl.Mat(),
            "right": sl.Mat(),
            "depth": sl.Mat(),
            "normal": sl.Mat(),
        }

    def _process_mats(mats):
        # Get CPU-side arrays and convert where needed.
        t_get_start = time.perf_counter()
        left_image = to_rgb(mats["left"].get_data())
        right_image = to_rgb(mats["right"].get_data())
        depth_ocv = mats["depth"].get_data()
        normal_map = mats["normal"].get_data()[..., :3]
        t_get_end = time.perf_counter()

        # Post-process: contiguous normal map and nan handling
        normal_map = np.ascontiguousarray(normal_map)  # keep uncommented for cmf

        # depth_ocv = np.where(np.isnan(depth_ocv), 0, depth_ocv)
        normal_map += 1
        normal_map /= 2.0
        # normal_map = np.where(np.isnan(normal_map), 0, normal_map)
        t_post_end = time.perf_counter()

        return (
            left_image,
            right_image,
            depth_ocv,
            normal_map,
            t_get_end - t_get_start,
            t_post_end - t_get_end,
        )

    run_params = sl.RuntimeParameters(
        enable_fill_mode=True, remove_saturated_areas=False
    )
    benchmark = StageBenchmark() if benchmarking else None
    last_report = time.perf_counter() if benchmarking else None

    # Double-buffering: acquire into one set while processing the other
    mats_curr = _alloc_mats()
    mats_next = _alloc_mats()

    executor = ThreadPoolExecutor(max_workers=1)
    prev_future = None
    prev_acq_time = None

    for acquisition_time in fps_limiter.rate(ZED_IMAGE_FPS):
        # Acquire a frame (grab + retrieves) into mats_curr, measuring total time
        retrieve_total = 0.0

        t_s = time.perf_counter()
        if zed.grab(run_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED grab error!")
        retrieve_total += time.perf_counter() - t_s

        t_s = time.perf_counter()
        zed.retrieve_image(mats_curr["left"], sl.VIEW.LEFT)
        retrieve_total += time.perf_counter() - t_s

        t_s = time.perf_counter()
        zed.retrieve_image(mats_curr["right"], sl.VIEW.RIGHT)
        retrieve_total += time.perf_counter() - t_s

        t_s = time.perf_counter()
        zed.retrieve_measure(mats_curr["depth"], sl.MEASURE.DEPTH)
        retrieve_total += time.perf_counter() - t_s

        t_s = time.perf_counter()
        zed.retrieve_measure(mats_curr["normal"], sl.MEASURE.NORMALS)
        retrieve_total += time.perf_counter() - t_s

        if benchmarking:
            benchmark.update_stage("retrieve", retrieve_total)

        # Kick off processing for the just-acquired mats.
        future_curr = executor.submit(_process_mats, mats_curr)

        # If we have a previous frame's processing, wait for it and yield results
        if prev_future is not None:
            (
                left_image,
                right_image,
                depth_ocv,
                normal_map,
                get_data_dur,
                postproc_dur,
            ) = prev_future.result()

            if benchmarking:
                benchmark.update_stage("get_data", get_data_dur)
                benchmark.update_stage("nan_to_num", postproc_dur)
                benchmark.update_fps()

            # Emit processed images for the previous acquisition time
            yield ZED_IMAGE_DIRECTION_LEFT, prev_acq_time, left_image
            yield ZED_IMAGE_DIRECTION_RIGHT, prev_acq_time, right_image
            yield ZED_DEPTH_DIRECTION, prev_acq_time, depth_ocv
            yield ZED_NORMAL_DIRECTION, prev_acq_time, normal_map

            if benchmarking:
                now = time.perf_counter()
                if now - last_report >= BENCH_REPORT_INTERVAL_S:  # type: ignore[arg-type]
                    print(f"[ZED] {benchmark.summary()}")
                    last_report = now

        # Rotate buffers and remember current as previous for the next loop
        prev_future = future_curr
        prev_acq_time = acquisition_time
        mats_curr, mats_next = mats_next, mats_curr


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
