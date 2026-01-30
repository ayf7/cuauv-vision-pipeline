#!/usr/bin/env python3
import os

from auv_build import ninja_common

build = ninja_common.Build("vision")

build.build_shared(
    "camera_message_framework",
    sources=[
        "lib/camera_message_framework_c.cpp",
        "lib/camera_message_framework.cpp",
        "lib/filelock.cpp",
    ],
    auv_deps=["auvlog", "fmt"],
    cflags=["-Ivision/", "-Ilib"],
)

build.build_shared(
    "capture_source",
    sources=[
        "lib/capture_source.cpp",
    ],
    auv_deps=["auvlog", "fmt", "camera_message_framework"],
    cflags=["-Ivision/", "-Ilib"],
)
build.build_shared(
    "auv-color-balance",
    ["utils/color_correction/color_balance.cpp"],
    cflags=[],
    pkg_confs=["opencv4"],
    auv_deps=["utils"],
)

# Python capture sources
build.install("auv-webcam-camera", f="vision/capture_sources/generic_camera.py")
build.install("auv-video-camera", f="vision/capture_sources/video.py")
build.install("auv-camera-stream-server", f="vision/capture_sources/stream_server.py")
build.install("auv-camera-stream-client", f="vision/capture_sources/stream_client.py")

# build zed on vehicle only
if os.environ["CUAUV_CONTEXT"] == "vehicle":
    build.build_cmd(
        "auv-zed-camera",
        sources=["capture_sources/zed.cpp"],
        auv_deps=[
            "auvlog",
            "fmt",
            "auvshm",
            "camera_message_framework",
            "capture_source",
        ],
        deps=["sl_zed", "cudart"],
        cflags=[
            "-Ivision/",
            "-Ilib",
            "-I/usr/local/zed/include",
            "-I/usr/local/cuda/include",
        ],
        lflags=[
            "-L/usr/local/zed/lib",
            "-Wl,-rpath,/usr/local/zed/lib",
            "-L/usr/local/cuda/lib64",
        ],
    )

build.install("auv-yolo-shm", f="vision/misc/yolo_shm.py")
build.install("auv-vision-runner", f="vision/runner.sh")
build.install("auv-vr", f="vision/runner.sh")

# only build flir on sub
if os.environ["CUAUV_CONTEXT"] == "vehicle":
    build.build_cmd(
        "auv-flir-camera",
        sources=["capture_sources/flir.cpp"],
        auv_deps=[
            "auvlog",
            "fmt",
            "auvshm",
            "camera_message_framework",
            "capture_source",
        ],
        deps=["Spinnaker"],
        cflags=["-Ivision/", "-Ilib", "-I/opt/spinnaker/include"],
    )
