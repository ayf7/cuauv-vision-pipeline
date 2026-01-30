/**
 * @file flir.cpp
 * @author Jeffrey Qian
 * @brief
 * @version 1.0
 * @date 2025-03-01
 */

#pragma once

#include <atomic>
#include <chrono>
#include <fmt/format.h>
#include <functional>
#include <iostream>
#include <optional>
#include <thread>
#include <vector>

#include "include/camera_message_framework.hpp"

namespace capture_source {
/// @brief A type alias for an atomic boolean flag used for signaling quit operations.
typedef std::atomic<bool> QuitFlag;

/// @brief limits the fps of a process
class FpsLimiter {
public:
    /**
     * @brief Constructs an FpsLimiter object with an optional frame rate.
     *
     * @param fps The target frame rate in frames per second. If not provided, no limit is enforced.
     */
    FpsLimiter(std::optional<uint32_t> fps = std::nullopt);

    /**
     * @brief Waits until the next frame should be processed based on the FPS limit.
     *
     * @return uint64_t Current time in milliseconds.
     */
    uint64_t tick();

private:
    uint64_t _target_frame_time_ms;
    std::chrono::high_resolution_clock::time_point _last_frame_time;
};

/// @brief handles processing threads for image capture
class CaptureSource {
public:
    CaptureSource();
    ~CaptureSource();

    /// @brief run image capture threads until they terminate
    void run_until_complete();

    /**
     * @brief Registers a user-defined function (UDL) to be executed in a separate thread. Note that
     * the first argument must be a reference to CaptureSource, and the second argument must be a
     * reference to QuitFlag
     *
     * @tparam Function The function type.
     * @tparam Args The argument types for the function.
     * @param name A name identifying the user-defined function.
     * @param func The function to be executed.
     * @param args Arguments to be passed to the function.
     * @example
     *
     * void flir_capture_udl(capture_source::CaptureSource &cs, capture_source::QuitFlag &quit_flag,
     * std::unique_ptr<Flir> flir);
     */
    template<typename Function, typename... Args>
    void register_udl(const std::string& name, Function func, Args&&... args) {
        // Ensure unique_ptrs are safely stored before moving
        using TupleType = std::tuple<std::decay_t<Args>...>;
        TupleType args_tuple(std::forward<Args>(args)...);

        _thread_handlers.emplace_back(
            [name, this, func = std::move(func), args_tuple = std::move(args_tuple)]() mutable {
                try {
                    std::apply(
                        [&](auto&&... unpacked_args) {
                            func(
                                std::ref(*this),
                                std::ref(_quit_flag),
                                std::forward<decltype(unpacked_args)>(unpacked_args)...
                            );
                        },
                        std::move(args_tuple)
                    );

                    bool ive_set = !_quit_flag;
                    _quit_flag.store(true);
                    if (ive_set) {
                        std::cout << fmt::format("Capture udl '{}' is exhausted.", name)
                                  << std::endl;
                    } else {
                        std::cout << fmt::format(
                            "Capture udl '{}' stopped as a result of another stop signal.",
                            name
                        ) << std::endl;
                    }
                } catch (const std::exception& exception) {
                    std::cerr << fmt::format("Caught exception '{}'", exception.what())
                              << std::endl;
                    _quit_flag.store(true);
                }
            }
        );
    }

    /**
     * @brief Writes an image frame to the specified direction.
     *
     * @tparam T The data type of the image.
     * @param direction The direction to which the image should be written.
     * @param acquisition_time The timestamp when the image was acquired in milliseconds.
     * @param width The width of the image in pixels.
     * @param height The height of the image in pixels.
     * @param depth The depth of the image (e.g., number of color channels).
     * @param data A pointer to the image data.
     */
    template<typename T>
    void write_image(
        const std::string& direction,
        const uint64_t acquisition_time,
        const size_t width,
        const size_t height,
        const size_t depth,
        const void* data
    ) {
        auto it = _block_handlers.find(direction);

        if (it == _block_handlers.end()) {
            auto [it_new, _] = _block_handlers.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(direction),
                std::forward_as_tuple(direction, width * height * depth * sizeof(T))
            );

            it = it_new;
        }

        it->second.write_frame(acquisition_time, width, height, depth, sizeof(T), data);
    }

    /**
     * @brief Writes multiple named planes into a single aggregated frame.
     * Each plane can have its own element size and dimensions.
     *
     * @param direction The block name to write into.
     * @param acquisition_time Timestamp in milliseconds.
     * @param planes A list of planes (width, height, depth, type_size, data, name).
     */
    void write_planes(
        const std::string& direction,
        const uint64_t acquisition_time,
        const std::vector<cmf::FramePlaneWrite>& planes
    ) {
        if (planes.empty())
            return;

        std::size_t total_bytes = 0;
        for (const auto& p: planes) {
            total_bytes += p.width * p.height * p.depth * p.type_size;
        }

        auto it = _block_handlers.find(direction);
        if (it == _block_handlers.end()) {
            auto [it_new, _] = _block_handlers.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(direction),
                std::forward_as_tuple(direction, total_bytes)
            );

            it = it_new;
        }

        it->second.write_frame(acquisition_time, planes.data(), planes.size());
    }

private:
    QuitFlag _quit_flag;
    std::vector<std::thread> _thread_handlers;
    std::unordered_map<std::string, cmf::Block> _block_handlers;
};
}; // namespace capture_source
