import sys
import enum
import time
from typing import Any, List, Tuple, Union, Optional, Sequence

import cffi
import numpy as np

from auv_python_helpers import get_library_path

ffi = cffi.FFI()

ffi.cdef(
    """
extern const char* BLOCK_STUB_CSTR;
extern int SUCCESS;
extern int NO_NEW_FRAME;
extern int FRAMEWORK_DELETED;

typedef struct Block Block;
typedef struct FramePlane {
    size_t width;
    size_t height;
    size_t depth;
    size_t type_size;
    size_t offset;
    char name[32];
} FramePlane;
typedef struct Frame {
    size_t width;
    size_t height;
    size_t depth;
    size_t type_size;
    uint64_t acquisition_time;
    uint64_t uid;
    void* data;
    size_t total_size;
    size_t plane_count;
    FramePlane planes[4];
} Frame;
typedef struct FramePlaneWrite {
    size_t width;
    size_t height;
    size_t depth;
    size_t type_size;
    const unsigned char* data;
    const char* name;
} FramePlaneWrite;
Block* create_block(const char* direction, const size_t max_entry_size_bytes);
Block* open_block(const char* direction);
void delete_block(Block* block);
int write_frame(Block* block,
				 uint64_t acquisition_time,
				 size_t width,
				 size_t height,
				 size_t depth,
				 size_t type_size,
				 const unsigned char* data);
int write_frame_planes(Block* block,
				 uint64_t acquisition_time,
				 const FramePlaneWrite* planes,
				 size_t plane_count);
int read_frame(Block* block, Frame* frame, bool block_thread);
Frame* create_frame();
void delete_frame(Frame* frame);
uint64_t frame_size(Frame* frame);
"""
)

_dllib: Any = ffi.dlopen(get_library_path("libcamera_message_framework.so"))


class ReadStatus(enum.Enum):
    """Enum wrapper for vision buffer read status"""

    SUCCESS = _dllib.SUCCESS  # type: ignore
    NO_NEW_FRAME = _dllib.NO_NEW_FRAME  # type: ignore
    FRAMEWORK_DELETED = _dllib.FRAMEWORK_DELETED  # type: ignore


class WriteStatus(enum.Enum):
    """Enum wrapper for vision buffer write status"""

    SUCCESS = _dllib.SUCCESS  # type: ignore
    FRAMEWORK_DELETED = _dllib.FRAMEWORK_DELETED  # type: ignore


BLOCK_STUB = ffi.string(_dllib.BLOCK_STUB_CSTR).decode()  # type: ignore


def encode_str(s: str):
    """
    Encodes a string into a numpy array of bytes (uint8).

    Parameters:
        s (str): The input string to be encoded.

    Returns:
        np.ndarray: A numpy array containing the encoded byte values of the string.
    """
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)


def decode_str(arr: np.ndarray):
    """
    Decodes a numpy array of bytes (uint8) into a string.

    Parameters:
        arr (np.ndarray): The input numpy array containing byte values to be decoded.

    Returns:
        str: The decoded string from the byte array.
    """
    return arr.tobytes().decode("utf-8")


class BlockAccessor:
    """A volatile memory-backed object (mmap-ed object) capable of being shared
    between multiple processes. Supports writes of numpy arrays up to 3-dimensions
    with underlying data type sizes that are 1, 4, or 8 bytes wide.
    """

    def __init__(
        self,
        direction: str,
        max_entry_size_bytes: Optional[int] = None,
        byte_type: type = np.uint8,
        short_type: type = np.float32,
        long_type: type = np.float64,
        block_thread: bool = False,
    ):
        """Initializes a BlockAccessor that will create/access the volatile-memory
        backed object within a context manager. The behavior of the accessor depends
        on the arguments passed into this initializer.

        Args:
            direction (str): the name given to the mmap object.
            max_entry_size_bytes (Optional[int], optional): bytes to allocate for a frame in the mmap-ed object; if left as None, the system will not allocate any memory, and will wait for the object to be created

            byte_type (type, optional): 1-byte wide data format from this block. Defaults to np.uint8.
            short_type (type, optional): 4-byte wide data format from this block. Defaults to np.float32.
            long_type (type, optional): 8-byte wide data format from this block. Defaults to np.float64.
        """

        assert (max_entry_size_bytes is None) or (
            max_entry_size_bytes > 0
        ), "max_entry_size_bytes, when specified, should be a positive integer"
        assert np.dtype(byte_type).itemsize == 1, "byte type must be 1 byte wide"
        assert np.dtype(short_type).itemsize == 4, "short type must be 4 bytes wide"
        assert np.dtype(long_type).itemsize == 8, "long type must be 8 bytes wide"

        self._direction = direction
        self._max_entry_size_bytes = max_entry_size_bytes
        self._type_lookup = {
            1: byte_type,
            4: short_type,
            8: long_type,
        }

        self._inside_ctx_manager = False
        self._block_ptr = ffi.NULL
        self._frame_ptr = ffi.NULL
        self._frame_data: Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]] = None
        self._last_plane_names: Tuple[str, ...] = tuple()
        self._block_thread: bool = block_thread
        self._acquisition_time: int = 0

    @property
    def direction(self) -> str:
        """Get name of the mmap-ed object"""
        return self._direction

    def block_thread(self) -> "BlockAccessor":
        """Implements the builder pattern. Allows read_frame to block the current thread
        when there is no new frame
        """
        self._block_thread = True
        return self

    def unblock_thread(self) -> "BlockAccessor":
        """Implements the builder pattern. Allows read_frame to return immediately if
        there is no new frame.
        """
        self._block_thread = False
        return self

    def write_frame(
        self,
        acquisition_time_ms: int,
        frame: Union[
            np.ndarray,
            Sequence[np.ndarray],
            Sequence[Tuple[str, np.ndarray]],
        ],
    ):
        """Write one or more numpy-backed planes into the mmap-ed object.

        Args:
            acquisition_time_ms (int): Time in milliseconds when the frame was acquired.
            frame (Union[np.ndarray, Sequence[np.ndarray]]): Either a single ndarray or a
                sequence of ndarrays representing individual planes.

        Raises:
            RuntimeError: If the accessor is not inside a context manager.
            RuntimeError: If a plane has unsupported dtype width or dimensionality.
            TypeError: If a non-ndarray object is provided.
            ValueError: If an empty plane sequence is provided.
        """

        if not self._inside_ctx_manager:
            raise RuntimeError(
                f"Attempted to access block while not in a context manager: {__file__}:{sys._getframe(1).f_lineno}"
            )

        plane_names: List[str] = []
        if isinstance(frame, np.ndarray):
            planes: List[np.ndarray] = [frame]
            plane_names = [""]
        elif isinstance(frame, Sequence):
            if len(frame) == 0:
                raise ValueError("empty frame sequence passed to write_frame")
            planes = []
            for idx, item in enumerate(frame):
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], np.ndarray)
                ):
                    plane_names.append(item[0])
                    planes.append(item[1])
                elif isinstance(item, np.ndarray):
                    plane_names.append("")
                    planes.append(item)
                else:
                    raise TypeError(
                        f"frame at index {idx} must be an ndarray or (name:str, ndarray)"
                    )
        else:
            raise TypeError("frame must be an ndarray or a sequence of ndarrays")

        contiguous_planes: List[np.ndarray] = []
        plane_dims: List[Tuple[int, int, int, int]] = []

        for idx, plane in enumerate(planes):
            contiguous = np.ascontiguousarray(plane)

            if contiguous.ndim == 0 or contiguous.ndim > 3:
                raise RuntimeError(
                    f"np.ndarray at index {idx} has {contiguous.ndim} dimensions, expected between 1-3"
                )

            if contiguous.itemsize not in self._type_lookup:
                raise RuntimeError(
                    f"np.ndarray at index {idx} has unsupported dtype width of {contiguous.itemsize} bytes"
                )

            height = contiguous.shape[0]
            width = contiguous.shape[1] if contiguous.ndim > 1 else 1
            depth = contiguous.shape[2] if contiguous.ndim > 2 else 1

            contiguous_planes.append(contiguous)
            plane_dims.append((height, width, depth, contiguous.itemsize))

        plane_count = len(contiguous_planes)
        plane_array = ffi.new("FramePlaneWrite[]", plane_count)
        buffers: List[Any] = []  # keep CFFI buffer objects alive for the call duration
        name_bufs: List[Any] = []

        for idx, contiguous in enumerate(contiguous_planes):
            height, width, depth, itemsize = plane_dims[idx]
            plane_array[idx].width = ffi.cast("size_t", width)
            plane_array[idx].height = ffi.cast("size_t", height)
            plane_array[idx].depth = ffi.cast("size_t", depth)
            plane_array[idx].type_size = ffi.cast("size_t", itemsize)

            buffer = ffi.from_buffer(contiguous)
            buffers.append(buffer)
            plane_array[idx].data = ffi.cast("const unsigned char*", buffer)
            # assign name if provided
            name = plane_names[idx] if idx < len(plane_names) else ""
            name_c = ffi.new("char[]", name.encode("utf-8"))
            name_bufs.append(name_c)
            plane_array[idx].name = name_c

        write_status = WriteStatus(
            _dllib.write_frame_planes(  # type: ignore
                self._block_ptr,
                ffi.cast("uint64_t", acquisition_time_ms),
                plane_array,
                ffi.cast("size_t", plane_count),
            )
        )

        return write_status

    def read_frame(
        self,
    ) -> Tuple[ReadStatus, Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]], int]:
        """Read the latest frame, if any, from the data segment in the mmap-ed object.
        If the block_thread property was set to true, this function may register itself as a
        watcher with a few second timeout to try and catch the latest frame.

        Raises:
            RuntimeError: Thrown when this function is not accessed in a context manager

        Returns:
            Tuple[ReadStatus, Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]], int]:
                ReadStatus, most recent frame payload (single ndarray or tuple of ndarrays, or None), acquisition time.
        """
        if not self._inside_ctx_manager:
            file = __file__
            frame = sys._getframe(1).f_lineno
            raise RuntimeError(
                f"Attempted to access block while not in a context manager: {file}:{frame}"
            )

        read_status = ReadStatus(
            _dllib.read_frame(self._block_ptr, self._frame_ptr, self._block_thread)
        )

        if read_status == ReadStatus.SUCCESS:
            acquisition_time: int = self._frame_ptr.acquisition_time  # type: ignore
            plane_count = int(self._frame_ptr.plane_count)  # type: ignore
            total_bytes = int(self._frame_ptr.total_size)  # type: ignore
            data = self._frame_ptr.data  # type: ignore

            if plane_count == 0 or total_bytes == 0:
                self._frame_data = None
                self._acquisition_time = acquisition_time
                self._last_plane_names = tuple()
                return read_status, self._frame_data, self._acquisition_time

            frame_buffer = ffi.buffer(data, total_bytes)
            planes: List[np.ndarray] = []
            names: List[str] = []

            for idx in range(plane_count):
                plane_meta = self._frame_ptr.planes[idx]  # type: ignore[index]
                width = int(plane_meta.width)
                height = int(plane_meta.height)
                depth = int(plane_meta.depth)
                itemsize = int(plane_meta.type_size)
                offset = int(plane_meta.offset)
                name = ffi.string(plane_meta.name).decode()
                names.append(name)

                dtype = self._type_lookup.get(itemsize)
                if dtype is None:
                    raise RuntimeError(
                        f"encountered unsupported type size {itemsize} while reading plane {idx}"
                    )

                plane_bytes = width * height * depth * itemsize
                if offset + plane_bytes > total_bytes:
                    raise RuntimeError(
                        f"plane {idx} with size {plane_bytes} at offset {offset} exceeds frame size {total_bytes}"
                    )
                plane_slice = frame_buffer[offset : offset + plane_bytes]

                planes.append(
                    np.frombuffer(plane_slice, dtype=dtype).reshape(
                        height, width, depth
                    )
                )

            self._acquisition_time = acquisition_time
            if plane_count == 1:
                self._frame_data = planes[0]
            else:
                self._frame_data = tuple(planes)
            self._last_plane_names = tuple(names)

        return read_status, self._frame_data, self._acquisition_time

    def last_plane_names(self) -> Tuple[str, ...]:
        return self._last_plane_names

    def __str__(
        self,
    ) -> str:
        type_str = ":".join(
            f"{size}->{dtype.__name__}"
            for size, dtype in sorted(self._type_lookup.items())
        )
        return f"Accessor(direction={self._direction}, datatypes={type_str})"

    def __enter__(self):
        """Use with context manager"""

        if self._inside_ctx_manager:
            raise RuntimeError(
                f"Double dip in context manager: {__file__}: {sys._getframe(1).f_lineno}"
            )

        cstr = ffi.new("char[]", self._direction.encode("utf8"))
        cstr_ptr = ffi.string(cstr)

        if self._max_entry_size_bytes is None:
            retried = False
            retry_count = 0
            self._block_ptr = _dllib.open_block(cstr_ptr)  # type: ignore
            while self._block_ptr == ffi.NULL:
                retry_count += 1

                print(
                    f"trying again to access {self._direction} in 1s, retry count={retry_count:<2}",
                    end="\r",
                    flush=True,
                )
                retried = True
                time.sleep(1)
                self._block_ptr = _dllib.open_block(cstr_ptr)  # type: ignore

            if retried:
                print(f"\nfound {self._direction}!!!", flush=True)

        else:
            self._block_ptr = _dllib.create_block(  # type: ignore
                cstr_ptr, ffi.cast("size_t", self._max_entry_size_bytes)
            )

            if self._block_ptr == ffi.NULL:
                raise RuntimeError(f"Failed to access {self._direction}")

        self._frame_ptr = _dllib.create_frame()  # type: ignore
        self._acquisition_time = 0
        self._frame_data = None
        self._inside_ctx_manager = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._block_ptr != ffi.NULL:
            _dllib.delete_block(self._block_ptr)  # type: ignore

        if self._frame_ptr != ffi.NULL:
            _dllib.delete_frame(self._frame_ptr)  # type: ignore

        self._block_ptr = ffi.NULL
        self._frame_ptr = ffi.NULL
        self._inside_ctx_manager = False
