import time
import signal
import threading
import traceback
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Callable,
    Optional,
    Sequence,
    Generator,
)

from numpy import ndarray

from auvlog.client import Logger, log as auvlog
from vision.core.bindings.camera_message_framework import BlockAccessor


class FpsLimiter:
    def __init__(self, name: str, quit_flag: threading.Event):

        fps_logger: Logger = getattr(auvlog.vision.capture_source.fps_limiter, name)

        self._slow = False
        self._logger = fps_logger
        self._quit_flag = quit_flag

    def rate(self, fps: Optional[int]):
        fps = fps if fps else 0
        assert fps >= 0, "given negative fps which is invalid"

        self._fps = fps
        self._target = 1.0 / fps if fps > 0 else 0
        self._last_time = 0
        return self

    def __iter__(self):
        self._last_time = time.monotonic()
        return self

    def __next__(self):
        if self._quit_flag.is_set():
            raise StopIteration

        current_time = time.monotonic()
        elapsed = current_time - self._last_time

        time_to_sleep = 0

        if elapsed < self._target:
            if self._slow:
                self._slow = False
                self._logger("recovered!", True)
            time_to_sleep = self._target - elapsed

        elif not self._slow:
            self._slow = True
            self._logger("too slow! dropped frames!", True)

        time.sleep(time_to_sleep)

        self._last_time = time.monotonic()
        return int(self._last_time * 1000)


class CaptureSource:
    """
    Base case for a capture source. This should never be directly created, but
    instead should be subclassed.
    """

    def __init__(self):
        """
        Initializes a capture source in the specified direction.

        Args:
            direction: direction, or block name in the camera message framework.
            fps: frames per second cap. persistent: if True, then the capture
                source will poll the subclass using the [acquire_next_image]
                method for frames to send, at a max rate of fps frames per
                second. If False, the subclass must take care of sending images
                itself.
        """
        name = self.__class__.__name__
        logger = auvlog.vision.capture_source

        self._logger: Logger = getattr(logger, name)
        self._frameworks: Dict[str, BlockAccessor] = {}
        self._threads: List[threading.Thread] = []
        self._quit_flag = threading.Event()

    def run_event_loop(self):
        def signal_handler(sig, frame):
            print("\n\nCtrl-C Caught")
            self._quit_flag.set()

        signal.signal(signal.SIGINT, signal_handler)

        for t in self._threads:
            t.start()

        while not self._quit_flag.is_set():
            time.sleep(0.1)

        for t in self._threads:
            t.join()

        self._logger(f"graceful shut down", True)

    def register_logical_udl(
        self,
        udl: Callable[[FpsLimiter, Tuple[Any, ...]], None],
        args: Tuple[Any, ...] = (),
    ):
        def callback():
            fps_limiter = FpsLimiter("", self._quit_flag)
            try:
                udl(fps_limiter, args)
            except Exception as e:
                self._logger(f"Caught exception printing stack trace and unwinding ...")
                traceback.print_exc()
                self._quit_flag.set()

        thread = threading.Thread(target=callback)
        self._threads.append(thread)

    def register_capture_udl(
        self,
        name: str,
        udl: Callable[
            [FpsLimiter, Tuple[Any, ...]],
            Generator[Tuple[str, int, Union[ndarray, Sequence[ndarray]]], None, None],
        ],
        args: Tuple[Any, ...] = (),
    ):
        def callback():
            self._logger(f"starting capture udl '{name}'", True)

            fps_limiter = FpsLimiter(name, self._quit_flag)

            try:
                for yielded in udl(fps_limiter, args):
                    # Support either (direction, time, frames) or (direction, time, frames, names)
                    if not isinstance(yielded, tuple):
                        raise RuntimeError(
                            f"capture UDL '{name}' yielded unsupported type {type(yielded)}"
                        )
                    if len(yielded) == 3:
                        direction, acquisition_time, img = yielded
                        names = None
                    elif len(yielded) == 4:
                        direction, acquisition_time, img, names = yielded
                    else:
                        raise RuntimeError(
                            f"capture UDL '{name}' yielded tuple of unexpected length {len(yielded)}"
                        )
                    self._send(direction, acquisition_time, img, names)
            except Exception as e:
                self._logger(
                    f"Caught exception in {name} printing stack trace and unwinding ..."
                )
                traceback.print_exc()
                self._quit_flag.set()

            ive_set = not self._quit_flag.is_set()
            self._quit_flag.set()

            if ive_set:
                self._logger(f"capture udl '{name}' exhausted", True)
            else:
                self._logger(
                    f"capture udl '{name}' stopped as a result of another stop signal",
                    True,
                )

        thread = threading.Thread(target=callback)
        self._threads.append(thread)

    def _send(
        self,
        direction: str,
        acquisition_time: int,
        img: Union[ndarray, Sequence[ndarray]],
        names: Optional[Sequence[str]] = None,
    ):
        if isinstance(img, ndarray):
            planes: Tuple[ndarray, ...] = (img,)
        elif isinstance(img, Sequence):
            if len(img) == 0:
                raise ValueError("capture source yielded an empty frame sequence")
            plane_list: List[ndarray] = []
            for idx, plane in enumerate(img):
                if not isinstance(plane, ndarray):
                    raise TypeError(
                        f"frame at index {idx} for direction '{direction}' is not an ndarray"
                    )
                plane_list.append(plane)
            planes = tuple(plane_list)
        else:
            raise TypeError(
                f"unsupported frame type {type(img)} for direction '{direction}'"
            )

        total_bytes = sum(int(plane.size * plane.itemsize) for plane in planes)
        if total_bytes <= 0:
            raise ValueError(
                f"total serialized size for direction '{direction}' must be positive"
            )

        if direction not in self._frameworks:
            self._frameworks[direction] = BlockAccessor(
                direction, max_entry_size_bytes=total_bytes
            )
            self._frameworks[direction].__enter__()

        # If names are provided, pair them up; otherwise send as unnamed planes
        payload: Union[
            ndarray,
            Tuple[ndarray, ...],
            Tuple[Tuple[str, ndarray], ...],
        ]
        if names is not None:
            if len(names) != len(planes):
                raise ValueError(
                    f"direction '{direction}' provided {len(planes)} planes but {len(names)} names"
                )
            payload = tuple((n, p) for n, p in zip(names, planes))
        else:
            payload = planes[0] if len(planes) == 1 else planes
        self._frameworks[direction].write_frame(acquisition_time, payload)

    def __del__(self):
        for accessors in self._frameworks.values():
            accessors.__exit__(None, None, None)
