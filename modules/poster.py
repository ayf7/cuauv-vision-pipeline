#!/usr/bin/env python3
import os

import numpy as np

import shm
from vision.core.base import ModuleBase, sources

try:
    from vision.capture_sources.zed import ZED_MAX_DISTANCE, ZED_MIN_DISTANCE
except ImportError:
    try:
        from vision.capture_sources.zed import ZED_MAX_DISTANCE, ZED_MIN_DISTANCE
    except ImportError:
        ZED_MIN_DISTANCE = 0
        ZED_MAX_DISTANCE = 1

import time

# Check if running in simulator mode
IS_SIMULATOR = os.environ.get("CUAUV_LOCALE") == "simulator"


class Poster(ModuleBase):

    @sources("zed[forward]", "zed[forward2]", "zed[depth]", "zed[normal]")
    def process_zed(self, img_forward, img_forward2, depth_img, normal_img):
        poster_status = shm.poster_status.get()

        poster_status.forward_counter += 1  # type: ignore
        self.post("forward", img_forward)
        self.post("forward2", img_forward2)

        # In simulator mode, depth and normal are already RGB uint8 placeholders
        # Skip scaling to avoid distortion
        if IS_SIMULATOR:
            depth_u8 = depth_img
            normal_u8 = normal_img
        else:
            # Real hardware: depth is float32, normal is float32 RGB in [0,1]
            depth_u8 = (depth_img - ZED_MIN_DISTANCE) / (
                ZED_MAX_DISTANCE - ZED_MIN_DISTANCE
            )
            depth_u8 = np.clip(depth_u8 * 255, 0, 255).astype(np.uint8)

            # normal: scale [0,1] -> [0,255] uint8
            normal_u8 = np.clip(normal_img * 255, 0, 255).astype(np.uint8)

        poster_status.depth_counter += 1  # type: ignore
        self.post("depth", depth_u8)
        poster_status.normal_counter += 1  # type: ignore
        self.post("normal", normal_u8)

        shm.poster_status.set(poster_status)

        # Metrics: end time and latency EMA (use alpha from zed_metrics)
        try:
            metrics = shm.zed_metrics.get()
            end_time = time.monotonic()
            metrics.end_time_sec = float(end_time)
            # Compute latency from capture's start_time_sec
            latency = end_time - float(metrics.start_time_sec)
            if 0.0 <= latency <= 10.0:
                alpha = float(metrics.alpha) if hasattr(metrics, "alpha") else 0.9
                one_minus = 1.0 - alpha
                prev = float(getattr(metrics, "latency_ema_sec", 0.0))
                metrics.latency_ema_sec = float(alpha * prev + one_minus * latency)
            # else: drop this calculation (wrap/overflow)
            shm.zed_metrics.set(metrics)
        except Exception:
            # do not fail the poster on SHM errors
            pass

    # Fallback for any uncovered sources (kept minimal)
    def process(self, direction, image):
        print(
            f"process fallback: direction={direction}, shape={image.shape if hasattr(image, 'shape') else 'N/A'}"
        )
        # Enable debug after first frame to see what's in cache
        if not self._debug_frame_cache:
            self._debug_frame_cache = True
            # Access the frame_cache through the loop
            import time
            import threading

            def debug_cache():
                time.sleep(2)  # Wait for some frames to accumulate
                # We can't easily access frame_cache here, but we can check what aliases we're getting
                print("Debug: Checking what directions we're receiving...")

            threading.Thread(target=debug_cache, daemon=True).start()
        pass


if __name__ == "__main__":
    Poster(["zed"])()
