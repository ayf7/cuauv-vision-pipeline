import time
from numpy import ndarray
from vision.core.base import ModuleBase, VideoSource
from vision.core.tuners import IntTuner, TunerBase, DoubleTuner

from typing import List

tuners: List[TunerBase] = [
    IntTuner("rgb_a",0, 0, 255),
    DoubleTuner("area_tuner", -23.23, -50, 2130)
]

class GateVision(ModuleBase):
    def process(self, direction: str, image: ndarray):
        self.post(f'post_{direction}', image)
        print(f'normalized (y, x) for {direction}', self.normalize((600, 800)))
        print(f'latency {direction}', self.get_latency())



GateVision(video_sources=['forward', 'downward'], tuners=tuners)()

