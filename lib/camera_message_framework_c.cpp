#include "include/camera_message_framework.hpp"
#include <auvlog/logger.h>
#include <chrono>
#include <filesystem>
#include <fmt/format.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

// used for garbage management because we cannot trust python.
//
// From testing, context managers do not free resources correctly, so
// using C++ RAII is an added layer of safety
static std::unordered_map<std::string, cmf::Block> cmf_heap;
static std::mutex global_lock;

extern "C" {

extern const char* BLOCK_STUB_CSTR = cmf::BLOCK_STUB.c_str();
extern const int SUCCESS = cmf::SUCCESS;
extern const int NO_NEW_FRAME = cmf::NO_NEW_FRAME;
extern const int FRAMEWORK_DELETED = cmf::FRAMEWORK_DELETED;

cmf::Block* create_block(const char* direction, const size_t max_entry_size_bytes) {
    std::scoped_lock lock { global_lock };
    std::string name { direction };
    std::unordered_map<std::string, cmf::Block>::iterator it = cmf_heap.find(name);

    if (it == cmf_heap.end()) {
        // not found, so need to create
        return &cmf_heap.emplace(name, std::move(cmf::Block(name, max_entry_size_bytes)))
                    .first->second;
    } else if (it->second.max_buffer_size() == max_entry_size_bytes) {
        // already created it so return the pointer
        return &it->second;
    } else {
        // already created, but max_entry_size_bytes does not match
        throw std::invalid_argument(fmt::format("duplicate allocation of {}", direction));
    }
}

cmf::Block* open_block(const char* direction) {
    std::scoped_lock lock { global_lock };
    std::string name { direction };
    std::unordered_map<std::string, cmf::Block>::iterator it = cmf_heap.find(name);

    if (it == cmf_heap.end()) {
        try {
            // not found, so need to create
            return &cmf_heap.emplace(name, std::move(cmf::Block(name))).first->second;
        } catch (std::filesystem::filesystem_error& e) {
            // allow python to handle this
            return nullptr;
        }
    } else {
        // already created it so return the pointer
        return &it->second;
    }
}

void delete_block(cmf::Block* block) {
    std::scoped_lock lock { global_lock };
    cmf_heap.erase(block->direction());
}

int write_frame(
    cmf::Block* block,
    std::uint64_t acquisition_time,
    std::size_t width,
    std::size_t height,
    std::size_t depth,
    std::size_t type_size,
    const unsigned char* data
) {
    return block->write_frame(acquisition_time, width, height, depth, type_size, data);
}

int write_frame_planes(
    cmf::Block* block,
    std::uint64_t acquisition_time,
    const cmf::FramePlaneWrite* planes,
    std::size_t plane_count
) {
    return block->write_frame(acquisition_time, planes, plane_count);
}

int read_frame(cmf::Block* block, cmf::Frame* frame, bool block_thread) {
    return block->read_frame(*frame, block_thread);
}

cmf::Frame* create_frame() {
    return new cmf::Frame();
}

void delete_frame(cmf::Frame* frame) {
    delete frame;
}

uint64_t frame_size(cmf::Frame* frame) {
    return frame->size();
}
}
