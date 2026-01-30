/**
 * @file zed.cpp
 * @author Anthony Fei
 * @brief ZED capture source in C++, featuring aggregated multi-plane frame.
 *
 * Mirrors the behavior of vision/capture_sources/zed.py:
 *  - Direction "zed" carries four named planes: "forward" (right),
 *    "forward2" (left), "depth" (F32), and "normal" (F32x3 in [0,1]).
 *  - A calibration thread updates camera settings from shm.camera_calibration.
 *
 * Empirically, zed.cpp's postprocessing is significantly faster with custom
 * vectorized operations + O3 compiling compared to off-the-shelf numpy
 * operations.
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "include/capture_source.hpp"
#include "libshm/c/shm.h"
#include <sl/Camera.hpp>

namespace fs = std::filesystem;

// Plane names: match Python implementation
static constexpr const char* ZED_PLANE_FORWARD = "forward"; // right cam
static constexpr const char* ZED_PLANE_FORWARD2 = "forward2"; // left cam
static constexpr const char* ZED_PLANE_DEPTH = "depth";
static constexpr const char* ZED_PLANE_NORMAL = "normal";

static constexpr const char* ZED_DIRECTION = "zed";

// Camera and processing params
static constexpr int ZED_CAMERA_FPS = 10;
static constexpr int ZED_IMAGE_FPS = 10;
static constexpr int ZED_DEPTH_FPS = 10;
static constexpr int ZED_NORMAL_FPS = 10;
static_assert(
    ZED_IMAGE_FPS == ZED_DEPTH_FPS && ZED_IMAGE_FPS == ZED_NORMAL_FPS,
    "Expect equal output fps"
);

static constexpr float ZED_MIN_DISTANCE = 0.5f; // meters
static constexpr float ZED_MAX_DISTANCE = 10.0f; // meters

struct ZedContext {
    std::shared_ptr<sl::Camera> cam;
};

// Drop alpha and keep channel order as RGBA->RGB (matching Python's cv2.COLOR_RGBA2RGB)
static void rgba_to_rgb(const sl::Mat& in_rgba, std::vector<unsigned char>& out_rgb) {
    const int w = in_rgba.getWidth();
    const int h = in_rgba.getHeight();
    out_rgb.resize(static_cast<size_t>(w) * h * 3);

    const sl::uchar4* src = in_rgba.getPtr<sl::uchar4>(sl::MEM::CPU);
    unsigned char* dst = out_rgb.data();
    const size_t px = static_cast<size_t>(w) * h;

    // NOTE: our C++ loops are compiled with -O3 and autovectorize to NEON on
    // Jetson, so they hit the same memory-bandwidth limit.
    for (size_t i = 0; i < px; ++i) {
        // sl::uchar4 exposes .r,.g,.b,.a (RGBA)
        dst[3 * i + 0] = src[i].r;
        dst[3 * i + 1] = src[i].g;
        dst[3 * i + 2] = src[i].b;
    }
}

static void normals_to_rgb01(const sl::Mat& in_normals, std::vector<float>& out_norm_rgb) {
    const int w = in_normals.getWidth();
    const int h = in_normals.getHeight();
    out_norm_rgb.resize(static_cast<size_t>(w) * h * 3);

    // Normals are sl::float4 (that is, 4 values of float32 - do not confuse with
    // 4-bit floating points numbers!) we keep xyz, map from [-1,1] to [0,1]
    const sl::float4* src = in_normals.getPtr<sl::float4>(sl::MEM::CPU);
    float* dst = out_norm_rgb.data();
    const size_t px = static_cast<size_t>(w) * h;

    // NOTE: our C++ loops are compiled with -O3 and autovectorize to NEON on
    // Jetson, so they hit the same memory-bandwidth limit.
    for (size_t i = 0; i < px; ++i) {
        dst[3 * i + 0] = (src[i].x + 1.f) * 0.5f;
        dst[3 * i + 1] = (src[i].y + 1.f) * 0.5f;
        dst[3 * i + 2] = (src[i].z + 1.f) * 0.5f;
    }
}

static void zed_capture_udl(
    capture_source::CaptureSource& cs,
    capture_source::QuitFlag& quit_flag,
    std::shared_ptr<ZedContext> ctx
) {
    capture_source::FpsLimiter limiter(ZED_IMAGE_FPS);
    sl::RuntimeParameters rt;
    rt.enable_depth = true;
    rt.enable_fill_mode = true;
    rt.remove_saturated_areas = false;

    sl::Mat left_rgba, right_rgba, depth_f32, normal_f32c4;

    std::vector<unsigned char> left_rgb;
    std::vector<unsigned char> right_rgb;
    std::vector<float> normals_rgb01;

    // Timing trackers
    bool have_last = false;
    std::chrono::steady_clock::time_point last_tp;

    while (!quit_flag.load()) {
        uint64_t acq_ms = limiter.tick();
        auto t_acq_start = std::chrono::steady_clock::now();
        if (ctx->cam->grab(rt) != sl::ERROR_CODE::SUCCESS) {
            continue;
        }

        // Retrieve into reusable Mats
        ctx->cam->retrieveImage(left_rgba, sl::VIEW::LEFT, sl::MEM::CPU);
        ctx->cam->retrieveImage(right_rgba, sl::VIEW::RIGHT, sl::MEM::CPU);
        ctx->cam->retrieveMeasure(depth_f32, sl::MEASURE::DEPTH, sl::MEM::CPU);
        ctx->cam->retrieveMeasure(normal_f32c4, sl::MEASURE::NORMALS, sl::MEM::CPU);
        auto t_acq_end = std::chrono::steady_clock::now();

        // Convert
        auto t_pp_start = std::chrono::steady_clock::now();
        rgba_to_rgb(left_rgba, left_rgb);
        rgba_to_rgb(right_rgba, right_rgb);
        normals_to_rgb01(normal_f32c4, normals_rgb01);
        auto t_pp_end = std::chrono::steady_clock::now();

        const int w = left_rgba.getWidth();
        const int h = left_rgba.getHeight();

        // Prepare planes. Depth is F32_C1; normals mapped to 0..1 with 3 channels
        cmf::FramePlaneWrite planes[4];

        planes[0] = cmf::FramePlaneWrite {
            static_cast<size_t>(w), static_cast<size_t>(h), 3u,
            sizeof(unsigned char),  left_rgb.data(),        ZED_PLANE_FORWARD2
        };
        planes[1] = cmf::FramePlaneWrite {
            static_cast<size_t>(w), static_cast<size_t>(h), 3u,
            sizeof(unsigned char),  right_rgb.data(),       ZED_PLANE_FORWARD
        };
        planes[2] = cmf::FramePlaneWrite { static_cast<size_t>(w),
                                           static_cast<size_t>(h),
                                           1u,
                                           sizeof(float),
                                           depth_f32.getPtr<float>(sl::MEM::CPU),
                                           ZED_PLANE_DEPTH };
        planes[3] = cmf::FramePlaneWrite {
            static_cast<size_t>(w), static_cast<size_t>(h), 3u,
            sizeof(float),          normals_rgb01.data(),   ZED_PLANE_NORMAL
        };

        // Update FPS EMA and SHM metrics (start time, acquisition/postprocess EMAs)
        auto now_tp = std::chrono::steady_clock::now();
        double inst_fps = 0.0;
        if (have_last) {
            double dt = std::chrono::duration<double>(now_tp - last_tp).count();
            if (dt > 0.0)
                inst_fps = 1.0 / dt;
        } else {
            have_last = true;
        }
        last_tp = now_tp;

        const double acq_sec = std::chrono::duration<double>(t_acq_end - t_acq_start).count();
        const double post_sec = std::chrono::duration<double>(t_pp_end - t_pp_start).count();
        const double total_sec = acq_sec + post_sec;
        const double acq_pct = total_sec > 0.0 ? (acq_sec / total_sec) * 100.0 : 0.0;
        const double post_pct = total_sec > 0.0 ? (post_sec / total_sec) * 100.0 : 0.0;

        try {
            const float alpha = shm->zed_metrics.g.alpha; // smoothing factor
            const float one_minus = 1.0f - alpha;

            // fps EMA using SHM alpha
            shm->zed_metrics.g.fps_ema =
                alpha * shm->zed_metrics.g.fps_ema + one_minus * static_cast<float>(inst_fps);

            // acquisition and postprocess EMAs (seconds)
            shm->zed_metrics.g.acquisition_time_ema_sec =
                alpha * shm->zed_metrics.g.acquisition_time_ema_sec
                + one_minus * static_cast<float>(acq_sec);
            shm->zed_metrics.g.postprocess_time_ema_sec =
                alpha * shm->zed_metrics.g.postprocess_time_ema_sec
                + one_minus * static_cast<float>(post_sec);

            // percent EMAs
            shm->zed_metrics.g.acquisition_time_ema_percent =
                alpha * shm->zed_metrics.g.acquisition_time_ema_percent
                + one_minus * static_cast<float>(acq_pct);
            shm->zed_metrics.g.postprocess_time_ema_percent =
                alpha * shm->zed_metrics.g.postprocess_time_ema_percent
                + one_minus * static_cast<float>(post_pct);

            // start time in monotonic seconds
            shm->zed_metrics.g.start_time_sec =
                static_cast<float>(std::chrono::duration<double>(now_tp.time_since_epoch()).count()
                );
        } catch (...) {
            // best-effort: ignore SHM write errors
        }

        cs.write_planes(
            ZED_DIRECTION,
            acq_ms,
            std::vector<cmf::FramePlaneWrite> { planes, planes + 4 }
        );
    }
}

static void zed_calib_udl(
    capture_source::CaptureSource& cs,
    capture_source::QuitFlag& quit_flag,
    std::shared_ptr<ZedContext> ctx
) {
    (void)cs;
    capture_source::FpsLimiter limiter(2);
    watcher_t watcher = create_watcher();
    shm_watch(camera_calibration, watcher);

    // Disable auto exposure/gain and auto white balance to use manual values
    ctx->cam->setCameraSettings(sl::VIDEO_SETTINGS::AEC_AGC, 0);
    ctx->cam->setCameraSettings(sl::VIDEO_SETTINGS::WHITEBALANCE_AUTO, 0);

    while (!quit_flag.load()) {
        limiter.tick();
        if (!watcher_has_changed(watcher)) {
            continue;
        }
        // Pull current settings
        const auto& c = shm->camera_calibration.g;
        ctx->cam->setCameraSettings(sl::VIDEO_SETTINGS::BRIGHTNESS, c.zed_brightness);
        ctx->cam->setCameraSettings(sl::VIDEO_SETTINGS::CONTRAST, c.zed_contrast);
        ctx->cam->setCameraSettings(sl::VIDEO_SETTINGS::HUE, c.zed_hue);
        ctx->cam->setCameraSettings(sl::VIDEO_SETTINGS::SATURATION, c.zed_saturation);
        ctx->cam->setCameraSettings(sl::VIDEO_SETTINGS::GAMMA, c.zed_gamma);
        ctx->cam->setCameraSettings(sl::VIDEO_SETTINGS::SHARPNESS, c.zed_sharpness);
        ctx->cam->setCameraSettings(
            sl::VIDEO_SETTINGS::WHITEBALANCE_TEMPERATURE,
            c.zed_white_balance
        );
        ctx->cam->setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, c.zed_exposure);
        ctx->cam->setCameraSettings(sl::VIDEO_SETTINGS::GAIN, c.zed_gain);
    }
    destroy_watcher(watcher);
}

int main() {
    shm_init();

    // Resolve ZED config file used by SDK
    const char* software_path = std::getenv("CUAUV_SOFTWARE");
    if (!software_path) {
        throw std::runtime_error("Please set environment variable 'CUAUV_SOFTWARE'");
    }
    const fs::path config_file =
        fs::path(software_path) / "vision/capture_sources/configs/zed.conf";

    // Open camera
    auto zed = std::make_shared<sl::Camera>();
    sl::InitParameters init_params;
    init_params.depth_mode = sl::DEPTH_MODE::NEURAL;
    init_params.optional_settings_path = sl::String(config_file.string().c_str());
    init_params.coordinate_units = sl::UNIT::METER;
    init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_params.depth_minimum_distance = ZED_MIN_DISTANCE;
    init_params.depth_maximum_distance = ZED_MAX_DISTANCE;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.camera_fps = ZED_CAMERA_FPS;

    auto status = zed->open(init_params);
    if (status != sl::ERROR_CODE::SUCCESS) {
        std::cerr << fmt::format("Failed to open ZED: {}\n", sl::toString(status).c_str())
                  << std::endl;
        return 1;
    }

    std::cout << "ZED camera initialized. Starting frame capture..." << std::endl;

    auto ctx = std::make_shared<ZedContext>();
    ctx->cam = zed;

    capture_source::CaptureSource cs;
    cs.register_udl("zed-capture", zed_capture_udl, ctx);
    cs.register_udl("zed-calibrate", zed_calib_udl, ctx);
    cs.run_until_complete();

    zed->close();
    return 0;
}
