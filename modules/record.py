#!/usr/bin/env python3

import os
from datetime import datetime
from typing import Tuple, Optional

import cv2
import numpy as np

import shm
from vision.core.base import ModuleBase, sources

try:
    from vision.capture_sources.zed import ZED_MAX_DISTANCE, ZED_MIN_DISTANCE
except ImportError:
    try:
        from vision.capture_sources.zed import ZED_MAX_DISTANCE, ZED_MIN_DISTANCE
    except ImportError:
        ZED_MIN_DISTANCE = 0.5
        ZED_MAX_DISTANCE = 10.0


TARGET_VIDEO_FPS = 10.0
LOG_DIR = os.path.join(os.environ.get("CUAUV_LOG", "/tmp"), "current")


class SimpleWriter:
    def __init__(self, filename: str, frame_size: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self._vw = cv2.VideoWriter(filename, fourcc, TARGET_VIDEO_FPS, frame_size)

    def write(self, frame: np.ndarray):
        if frame.ndim == 2:
            frame = cv2.merge((frame, frame, frame))
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.merge((frame[:, :, 0],) * 3)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported frame shape {frame.shape}")
        self._vw.write(frame)

    def close(self):
        self._vw.release()


class Record(ModuleBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._left_writer: Optional[SimpleWriter] = None
        self._right_writer: Optional[SimpleWriter] = None
        self._depth_writer: Optional[SimpleWriter] = None
        self._normal_writer: Optional[SimpleWriter] = None
        self._open_dir: Optional[str] = None

    def _ensure_open(
        self, left: np.ndarray, right: np.ndarray, depth: np.ndarray, normal: np.ndarray
    ):
        if self._left_writer is not None and self._right_writer is not None:
            return

        active_mission = shm.active_mission.get()
        log_dir = (
            LOG_DIR
            if not active_mission.log_path
            else active_mission.log_path.decode("utf-8")
        )
        os.makedirs(log_dir, exist_ok=True)
        self._open_dir = log_dir

        ts = datetime.today().strftime("%Y%m%d_%H%M%S")
        left_name = os.path.join(log_dir, f"zed_left_{ts}.mp4")
        right_name = os.path.join(log_dir, f"zed_right_{ts}.mp4")
        depth_name = os.path.join(log_dir, f"zed_depth_{ts}.mp4")
        normal_name = os.path.join(log_dir, f"zed_normal_{ts}.mp4")

        left_size = (left.shape[1], left.shape[0])
        right_size = (right.shape[1], right.shape[0])
        depth_size = (depth.shape[1], depth.shape[0])
        normal_size = (normal.shape[1], normal.shape[0])

        self._left_writer = SimpleWriter(left_name, left_size)
        self._right_writer = SimpleWriter(right_name, right_size)
        self._depth_writer = SimpleWriter(depth_name, depth_size)
        self._normal_writer = SimpleWriter(normal_name, normal_size)

    def _close_all(self):
        if self._left_writer:
            self._left_writer.close()
        if self._right_writer:
            self._right_writer.close()
        if self._depth_writer:
            self._depth_writer.close()
        if self._normal_writer:
            self._normal_writer.close()
        self._left_writer = self._right_writer = self._depth_writer = (
            self._normal_writer
        ) = None
        self._open_dir = None

    @sources("zed[forward]", "zed[forward2]", "zed[depth]", "zed[normal]")
    def record_zed(
        self, left: np.ndarray, right: np.ndarray, depth: np.ndarray, normal: np.ndarray
    ):
        # Depth: scale meters to 0..255 for display/record
        depth_u8 = (depth - ZED_MIN_DISTANCE) / (ZED_MAX_DISTANCE - ZED_MIN_DISTANCE)
        depth_u8 = np.clip(depth_u8, 0.0, 1.0)
        depth_u8 = (depth_u8 * 255).astype(np.uint8)

        # Normal: ensure 3 channels; [0,1] to 0..255
        if normal.ndim == 3 and normal.shape[2] >= 3:
            normal = normal[:, :, :3]
        normal_u8 = np.clip(normal * 255.0, 0, 255).astype(np.uint8)

        self.post("zed_left", left)
        self.post("zed_right", right)
        self.post("zed_depth", depth_u8)
        self.post("zed_normal", normal_u8)

        active_mission = shm.active_mission.get()
        if not active_mission.active:
            if self._left_writer is not None or self._right_writer is not None:
                self._close_all()
            return

        # Ensure writers are open
        self._ensure_open(left, right, depth, normal)

        # Write frames once per bundle (ignore duplication)
        self._left_writer.write(left)  # type: ignore
        self._right_writer.write(right)  # type: ignore
        self._depth_writer.write(depth_u8)  # type: ignore
        self._normal_writer.write(normal_u8)  # type: ignore

    # Fallback no-op process
    def process(self, direction, image):
        pass


if __name__ == "__main__":
    # Subscribe to the aggregated ZED source; writers record synchronized frames per bundle.
    Record(["zed"])()
