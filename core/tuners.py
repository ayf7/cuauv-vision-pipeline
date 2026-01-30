import struct
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Callable, List, Any
from vision.core.bindings.camera_message_framework import BlockAccessor
MAX_OPTION_SIZE_BYTE = 256

T = TypeVar('T')


class TunerBase(ABC, Generic[T]):
    def __init__(self, name: str, default_value: T):
        assert name.count(' ') == 0, f"Tuner name '{name}' cannot have spaces"
        assert name.count('/') == 0, f"Tuner name '{name}' cannot have slashes"
        self._name = name
        self._current_value = default_value

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, value: object) -> bool:
        if isinstance(value, self.__class__):
            return self._name == value._name
        return False

    def __str__(self) -> str:
        return f'{self.__class__.__name__}_{self._name}'

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._current_value

    @abstractmethod
    def byte_size(self) -> int:
        raise NotImplementedError('_TunerBase.byte_size')

    @abstractmethod
    def serialize(self) -> bytes:
        raise NotImplementedError('_TunerBase.serialize')

    @abstractmethod
    def deserialize(self, buffer: bytes):
        raise NotImplementedError('_TunerBase.deserialize')


class IntTuner(TunerBase[int]):
    def __init__(self,
                 name: str,
                 default_value: int,
                 min_value: int = 0,
                 max_value: int = 255,
                 validator: Callable[[int], bool] = lambda x: True
                 ):

        assert min_value <= max_value, f'min value = {min_value} is not leq to max value = {max_value}'
        super(IntTuner, self).__init__(name, default_value)

        self._min_value = min_value
        self._max_value = max_value
        self._packing_format = f'{len(self._name)}siii'
        self._validator = lambda x: validator(
            x) and min_value <= x <= max_value

    def byte_size(self) -> int:
        return struct.calcsize(self._packing_format)

    def serialize(self) -> bytes:
        return struct.pack(self._packing_format, self._name.encode(), self._current_value, self._min_value, self._max_value)

    def deserialize(self, buffer: bytes):
        name, current_value, self._min_value, self._max_value = struct.unpack(
            self._packing_format, buffer)

        self._name = name.decode()
        if self._validator(current_value):
            self._current_value = current_value


class DoubleTuner(TunerBase[float]):
    def __init__(self,
                 name: str,
                 default_value: float,
                 min_value: float = -10_000,
                 max_value: float = 10_000,
                 validator: Callable[[float], bool] = lambda x: True
                 ):

        assert min_value <= max_value, f'min value = {min_value} is not leq to max value = {max_value}'
        super(DoubleTuner, self).__init__(name, default_value)

        self._min_value = min_value
        self._max_value = max_value
        self._packing_format = f'{len(self._name)}sddd'
        self._validator = lambda x: validator(
            x) and min_value <= x <= max_value

    def byte_size(self) -> int:
        return struct.calcsize(self._packing_format)

    def serialize(self) -> bytes:
        return struct.pack(self._packing_format, self._name.encode(), self._current_value, self._min_value, self._max_value)

    def deserialize(self, buffer: bytes):
        name, current_value, self._min_value, self._max_value = struct.unpack(
            self._packing_format, buffer)

        self._name = name.decode()
        if self._validator(current_value):
            self._current_value = current_value


class BoolTuner(TunerBase[bool]):
    def __init__(self,
                 name: str,
                 default_value: bool,
                 ):

        super(BoolTuner, self).__init__(name, default_value)

        self._packing_format = f'{len(self._name)}s?'

    def byte_size(self) -> int:
        return struct.calcsize(self._packing_format)

    def serialize(self) -> bytes:
        return struct.pack(self._packing_format, self._name.encode(), self._current_value)

    def deserialize(self, buffer: bytes):
        name, current_value = struct.unpack(self._packing_format, buffer)

        self._name = name.decode()
        self._current_value = current_value


def tuner_from_bytes(name: str, data: bytes):
    ret = IntTuner('hello', 0)
    ret.deserialize(data)
    return ret
    # if 'IntTuner' in name:
    #     ret = IntTuner('', 0)
    #     ret.deserialize(data)
    #     return ret
    # elif 'DoubleTuner' in name:
    #     ret = DoubleTuner('', 0)
    #     ret.deserialize(data)
    #     return ret
    # elif 'BoolTuner' in name:
    #     ret = BoolTuner('', False)
    #     ret.deserialize(data)
    #     return ret
    # else:
    #     raise ValueError(f'{name} is not a valid tuner')

