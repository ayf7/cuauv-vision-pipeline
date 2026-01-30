# CUAUV: Vision Pipeline Architecture

Cornell University Autonomous Underwater Vehicle (CUAUV, cuauv.org) is an undergraduate robotics team consisting of mechanical, electrical, and software engineers that designs and builds an autonomous submarine every year to compete in the RoboSub competition. This repository contains the source code for a **custom computer vision pipeline** we've implemented to support robust processing of different various image sources to one or more computer vision models that may request them.

![Vision pipeline](assets/diagram.png)

[This link](https://docs.google.com/presentation/d/1-rNQczGqGKQJ9OqOSxplmhVOo_KmkaavHkseQvQlJok/edit?usp=sharing) contains a presentation that goes through the motivation and architectural details of this system.

## Key Components

The vision pipeline consists of the following components:

* **Capture sources** (capture_sources/)- classes used to interface with camera-specific APIs for acquiring streaming images.

* **Camera messaging framework** (lib/, include/) - a C++ shared memory buffer implementing a seqlock for lock-free, zero-copy inter-process frame passing between capture sources and vision modules. Capture sources write frames into the buffer and vision modules read from it without blocking writers, enabling concurrent image acquisition and processing.

* **Vision modules** (modules/) - client-facing implementation logic for computer vision models. Each module inherits from `ModuleBase` and implements a `process()` method that receives a camera direction and image frame. Modules handle detection, segmentation, and pose estimation for competition-specific tasks (e.g. buoy detection, bin detection, YOLO-based object detection) and post results to shared memory for use by the vehicle's control system.

* **Handlers** (handlers/) - post-processing hooks that attach to vision modules (primarily the YOLO module) to perform task-specific interpretation of model outputs, such as oriented bounding box detection for torpedoes, gate localization, and slalom gate processing.
