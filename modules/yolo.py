#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Union, Callable

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

import shm
from vision.core import tuners
from vision.core.handlers import HandlerMixin
from vision.core.base import ModuleBase, sources
from vision.handlers.torpedoes import TorpedoesOBB
from vision.yolo.data import MAP_FN, OBBData, PoseData, YOLOData

YOLO_WEIGHT = "obb_v14.pt"

HANDLERS = [
    TorpedoesOBB("torpedoes"),
]

TUNERS = [
    tuners.DoubleTuner("torpedo_threshold", 0.1, 0, 1),
    tuners.DoubleTuner("slalom_threshold", 0.0, 0, 1),
    tuners.DoubleTuner("gate_threshold", 0.1, 0, 1),
    tuners.DoubleTuner("gate_behind_threshold", 0.7, 0, 1),
    tuners.DoubleTuner("bins_threshold", 0.4, 0, 1),
    tuners.DoubleTuner("manipulator_threshold", 0.4, 0, 1),
]

# just a type alias
DetectionData = Union[YOLOData, OBBData, PoseData]


# TODO: HandlerMixin should be an inherited property of [ModuleBase].
class Yolo(ModuleBase, HandlerMixin):

    def __init__(self, video_sources, tuners, handlers, **kwargs):
        # TODO: convert this into super()
        ModuleBase.__init__(self, video_sources, tuners, **kwargs)
        HandlerMixin.__init__(self, handlers)

        self.weight_path = (
            Path("/home/software/cuauv/workspaces/yolo_weights") / YOLO_WEIGHT
        )
        """Path to weight that is loaded."""

        self.model = YOLO(str(self.weight_path))
        self.device = "cpu" if os.environ["CUAUV_LOCALE"] == "simulator" else "cuda"
        self.model.to(self.device)
        """The YOLO model, via PyTorch."""

        self.yolo_model_type = self.model.task
        """Type of model, between [detect], [obb], and [pose] for now."""

        self.map_fn: Callable[[dict], DetectionData] = MAP_FN[self.yolo_model_type]
        """Maps model type to the appropriate mapping function (mapping_utils.py)"""

        print("YOLO MODEL INITIALIZED:")
        print(f"weight: \t{self.weight_path}")
        print(f"device: \t{self.device}")
        print(f"model type: \t{self.yolo_model_type}")

    def torpedoes_active(self) -> bool:
        return shm.active_objects.yolo_torpedoes_board.get()

    def torpedoes_direction(self, direction: str) -> str:
        return shm.active_objects.yolo_torpedoes_board_direction.get() == direction

    # def slalom_active(self) -> bool:
    #     return shm.active_objects.yolo_slalom.get()

    # def slalom_direction(self, direction: str) -> str:
    #     return shm.active_objects.yolo_slalom_direction.get() == direction

    # def gate_active(self) -> bool:
    #     return shm.active_objects.yolo_gate.get()

    # def gate_direction(self, direction: str) -> str:
    #     return shm.active_objects.yolo_gate_direction.get() == direction

    # def gate_behind_active(self) -> bool:
    #     return shm.active_objects.yolo_gate_behind.get()

    # def gate_behind_direction(self, direction: str) -> str:
    #     return shm.active_objects.yolo_gate_behind_direction.get() == direction

    # def bins_active(self) -> bool:
    #     return shm.active_objects.yolo_bins.get()

    # def bins_direction(self, direction: str) -> str:
    #     return shm.active_objects.yolo_bins_direction.get() == direction

    # def manipulator_active(self) -> bool:
    #     return shm.active_objects.yolo_manipulator.get()

    # def manipulator_direction(self, direction: str) -> str:
    #     return shm.active_objects.yolo_manipulator_direction.get() == direction

    @sources("zed[forward]")
    def fwd_process(self, image: np.ndarray):
        """
        The YOLO process function is in charge of running model inference on the
        image, then mapping each image to their respective object handlers.

        Object handlers are an abstraction of AsyncBase.process, but includes
        the YOLO model information to go with it.
        """
        direction = "forward"  # TODO: LEGACY CODE
        self.post("original image", image)
        results = self.model.track(image, verbose=False)[
            0
        ]  # We don't batch, so there should only be one result
        results = results.summary()

        # NOTE: the following provides explicit mapping rules from each YOLO object identifier
        # to the name of each handler. Feel free to modify this logic however you want.
        if self.torpedoes_active():
            torpedoes_info = {"torpedo_board": [], "shark_hole": [], "saw_hole": []}

        for result in results:
            result_data: DetectionData = self.map_fn(result)

            # Matching
            match result_data.name:
                case "torpedo_board" | "shark_hole" | "saw_hole":
                    if self.torpedoes_active() and self.torpedoes_direction(direction):
                        torpedoes_info[result_data.name].append(result_data)
                # case "pole_red" | "pole_white":
                #     if self.slalom_active() and self.slalom_direction(direction):
                #         slalom_info[result_data.name].append(result_data)
                # case "shark" | "saw":
                #     if self.gate_active() and self.gate_direction(direction):
                #         gate_info[result_data.name].append(result_data)
                # case "gate_behind":  # NEW: roboflow class name
                #     if self.gate_behind_active() and self.gate_behind_direction(
                #         direction
                #     ):
                #         gate_behind_info["gate_behind"].append(result_data)
                # case "bin_shark" | "bin_saw" | "bin":
                #     if self.bins_active() and self.bins_direction(direction):
                #         bins_info[result_data.name].append(result_data)
                # case "spoon" | "cup" | "pink_basket" | "yellow_basket":
                #     if self.manipulator_active() and self.manipulator_direction(
                #         direction
                #     ):
                #         manip_info[result_data.name].append(result_data)
                case _:
                    pass
                    # print(f"Unhandled YOLO class: {result_data.name}")

        # Calling each handler, if active

        if self.torpedoes_direction(direction):
            if self.torpedoes_active():
                self.handlers["torpedoes"].process(
                    direction,
                    image.copy(),
                    torpedoes_info["torpedo_board"],
                    torpedoes_info["shark_hole"],
                    torpedoes_info["saw_hole"],
                )
            else:
                self.handlers["torpedoes"].post_grayscale(image)

        # if self.slalom_direction(direction):
        #     if self.slalom_active():
        #         self.handlers["slalom"].process(
        #             direction,
        #             image.copy(),
        #             slalom_info["pole_red"],
        #             slalom_info["pole_white"],
        #         )
        #     else:
        #         self.handlers["slalom"].post_grayscale(image)

        # if self.gate_direction(direction):
        #     if self.gate_active():
        #         self.handlers["gate"].process(
        #             direction,
        #             image.copy(),
        #             gate_info["shark"],
        #             gate_info["saw"],
        #         )
        #     else:
        #         self.handlers["gate"].post_grayscale(image)

        # if self.gate_behind_direction(direction):
        #     if self.gate_behind_active():
        #         self.handlers["gate_behind"].process(
        #             direction,
        #             image.copy(),
        #             gate_behind_info["gate_behind"],
        #         )
        #     else:
        #         self.handlers["gate_behind"].post_grayscale(image)

        # if self.bins_direction(direction):
        #     if self.bins_active():
        #         self.handlers["bins"].process(
        #             direction,
        #             image.copy(),
        #             bins_info["bin_shark"],
        #             bins_info["bin_saw"],
        #             bins_info["bin"],
        #         )
        #     else:
        #         self.handlers["bins"].post_grayscale(image)

        # if self.bins_direction(direction):
        #     if self.bins_active():
        #         self.handlers["bins"].process(
        #             direction,
        #             image.copy(),
        #             bins_info["bin_shark"],
        #             bins_info["bin_saw"],
        #             bins_info["bin"],
        #         )
        #     else:
        #         self.handlers["bins"].post_grayscale(image)
        # if self.manipulator_direction(direction):
        #     if self.manipulator_active():
        #         self.handlers["manipulator"].process(
        #             direction,
        #             image.copy(),
        #             manip_info["spoon"],
        #             manip_info["cup"],
        #             manip_info["pink_basket"],
        #             manip_info["yellow_basket"],
        #         )
        #     else:
        #         self.handlers["manipulator"].post_grayscale(image)


if __name__ == "__main__":
    Yolo(video_sources=["zed"], tuners=TUNERS, handlers=HANDLERS)()
