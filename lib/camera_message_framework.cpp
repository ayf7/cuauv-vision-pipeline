#include "include/camera_message_framework.hpp"
#include "include/filelock.hpp"

#include <atomic>
#include <auvlog/logger.h>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fmt/format.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

namespace cmf {

struct PlaneMetadata {
    std::uint64_t width;
    std::uint64_t height;
    std::uint64_t depth;
    std::uint64_t type_size;
    std::uint64_t offset;
    char name[PLANE_NAME_MAX_LEN];
};

struct FrameMetadata {
    std::atomic<uint64_t> v_a, v_b;
    std::uint64_t acquisition_time;
    std::uint64_t total_size;
    std::uint64_t width;
    std::uint64_t height;
    std::uint64_t depth;
    std::uint64_t type_size;
    std::uint64_t plane_count;
    PlaneMetadata planes[MAX_PLANE_CNT];
};

struct Buffer {
public:
    Buffer() = delete;

public:
    std::atomic<uint64_t> uid;
    std::size_t max_entry_size_bytes;

    bool deleted;
    FrameMetadata metadata[BUFFER_CNT];

    pthread_cond_t cond;
    pthread_mutex_t cond_mutex;

    alignas(64) unsigned char data[];
};

///////////////////////////////////////////////////////////////////////////////
/// Helpers
///
///////////////////////////////////////////////////////////////////////////////

std::string filename_from_direction(const std::string& direction) {
    // Check if "/"  exists in direction. Note that "/" is the ONLY forbidden 8
    // bit character in Linux filenames
    if (direction.find('/') != std::string::npos)
        throw std::invalid_argument("Block name contains '/' which is forbidden."); // TODO
    return BLOCK_STUB + direction;
}

inline const std::size_t shm_size(const Buffer* buffer) noexcept {
    return sizeof(Buffer) + buffer->max_entry_size_bytes * BUFFER_CNT;
}

///////////////////////////////////////////////////////////////////////////////
/// Buffer
///////////////////////////////////////////////////////////////////////////////
Buffer* create_block(int fd, std::size_t max_entry_size, const Block& b) {
    const std::size_t required_bytes = sizeof(Buffer) + max_entry_size * BUFFER_CNT;

    if (ftruncate(fd, required_bytes) == -1) {
        return nullptr;
    }

    const int prot_flags = PROT_READ | PROT_WRITE;
    void* raw_memory = mmap(NULL, lseek(fd, 0, SEEK_END), prot_flags, MAP_SHARED, fd, 0);

    if (raw_memory == (void*)(-1)) {
        return nullptr;
    }

    Buffer* buffer = (Buffer*)raw_memory;

    for (std::size_t i = 0; i < BUFFER_CNT; i++) {
        buffer->metadata[i].v_a = 0;
        buffer->metadata[i].v_b = 0;

        buffer->metadata[i].acquisition_time = 0;
        buffer->metadata[i].total_size = 0;
        buffer->metadata[i].width = 0;
        buffer->metadata[i].height = 0;
        buffer->metadata[i].depth = 0;
        buffer->metadata[i].type_size = 0;
        buffer->metadata[i].plane_count = 0;

        for (std::size_t j = 0; j < MAX_PLANE_CNT; j++) {
            PlaneMetadata pm { 0, 0, 0, 0, 0, { 0 } };
            pm.name[0] = '\0';
            buffer->metadata[i].planes[j] = pm;
        }
    }

    buffer->max_entry_size_bytes = max_entry_size;
    buffer->uid = 0;
    buffer->deleted = false;

    pthread_condattr_t attrcond;
    pthread_condattr_init(&attrcond);
    pthread_condattr_setpshared(&attrcond, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&buffer->cond, &attrcond);

    pthread_mutexattr_t attrmutex;
    pthread_mutexattr_init(&attrmutex);
    pthread_mutexattr_setpshared(&attrmutex, PTHREAD_PROCESS_SHARED);
    pthread_mutexattr_setrobust(&attrmutex, PTHREAD_MUTEX_ROBUST);
    pthread_mutex_init(&buffer->cond_mutex, &attrmutex);

    auvlog_info(
        fmt::format("Created block at '{}' with size {} bytes", b.filename(), shm_size(buffer))
    );
    return buffer;
}

// Global lock guarantees buffer is not being destoryed
Buffer* open_block(int fd, const Block& b) {
    const int prot_flags = PROT_READ | PROT_WRITE;
    void* raw_memory = mmap(NULL, lseek(fd, 0, SEEK_END), prot_flags, MAP_SHARED, fd, 0);
    if (raw_memory == (void*)-1) {
        return nullptr;
    }

    Buffer* buffer = (Buffer*)raw_memory;

    auvlog_info(
        fmt::format("Opened block at '{}' with size {} bytes", b.filename(), shm_size(buffer))
    );

    return buffer;
}

Block::Block(const std::string& direction, const size_t max_entry_size) {
    filelock::Filelock master_lock(GLOBAL_LOCK);
    std::string filename = filename_from_direction(direction);

    bool file_exists = access(filename.c_str(), F_OK) == 0;
    int fd = open(filename.c_str(), O_RDWR | O_CREAT, S_IRWXU);

    if (fd == -1) {
        throw std::system_error(errno, std::generic_category(), filename);
    }

    _creator = true;
    _direction = direction;
    _filename = filename;
    _buffer = file_exists ? open_block(fd, *this) : create_block(fd, max_entry_size, *this);

    if (close(fd) == -1) {
        throw std::system_error(errno, std::generic_category(), filename);
    }

    if (_buffer == nullptr) {
        remove(filename.c_str());
        throw std::system_error(errno, std::generic_category(), filename);
    }

    if (_buffer->max_entry_size_bytes != max_entry_size) {
        remove(filename.c_str());
        throw std::invalid_argument(fmt::format(
            "Opened existing block named '{}' and got size mismatch: {} != {} bytes",
            _filename,
            max_entry_size,
            _buffer->max_entry_size_bytes
        ));
    }
}

Block::Block(const std::string& direction) {
    filelock::Filelock master_lock(GLOBAL_LOCK);

    std::string filename = filename_from_direction(direction);
    if (!std::filesystem::exists(filename)) {
        throw std::filesystem::filesystem_error(
            "Block does not exist",
            filename,
            std::make_error_code(std::errc::no_such_file_or_directory)
        );
    }

    int fd = open(filename.c_str(), O_RDWR, S_IRWXU);

    if (fd == -1) {
        throw std::system_error(errno, std::generic_category(), filename);
    }

    _creator = false;
    _direction = direction;
    _filename = filename;
    _buffer = open_block(fd, *this);
    close(fd);
}

Block::Block(Block&& other) noexcept {
    _filename = std::move(other._filename);
    _direction = std::move(other._direction);
    _buffer = other._buffer;
    _creator = other._creator;
    other._buffer = nullptr;
}

Block& Block::operator=(Block&& other) {
    if (this != &other) {
        _filename = std::move(other._filename);
        _direction = std::move(other._direction);
        _buffer = other._buffer;
        _creator = other._creator;
        other._buffer = nullptr;
    }

    return *this;
}

Block::~Block() {
    if (_buffer == nullptr) {
        // null if from move constructor
        return;
    }
    if (_creator) {
        _buffer->deleted = true;
        remove(_filename.c_str());
        auvlog_info(fmt::format("Destroyed block at '{}' and freed {} bytes", _filename, shm_size())
        );
    }

    munmap(_buffer, shm_size());
}

int Block::write_frame(
    std::uint64_t acquisition_time,
    std::size_t width,
    std::size_t height,
    std::size_t depth,
    std::size_t type_size,
    const void* bytes
) const noexcept {
    FramePlaneWrite plane { width, height, depth, type_size, bytes };
    return write_frame(acquisition_time, &plane, 1);
}

int Block::write_frame(
    std::uint64_t acquisition_time,
    const FramePlaneWrite* planes,
    std::size_t plane_count
) const noexcept {
    if (_buffer->deleted) [[unlikely]] {
        return FRAMEWORK_DELETED;
    }

    if (planes == nullptr) [[unlikely]] {
        throw std::invalid_argument("planes pointer cannot be null");
    }

    if (plane_count == 0 || plane_count > MAX_PLANE_CNT) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "invalid plane count {}, expected between 1 and {}",
            plane_count,
            MAX_PLANE_CNT
        ));
    }

    std::size_t entry_size = 0;

    for (std::size_t i = 0; i < plane_count; i++) {
        const auto& plane = planes[i];
        if (plane.data == nullptr) [[unlikely]] {
            throw std::invalid_argument(fmt::format("plane {} has null data pointer", i));
        }

        const std::size_t plane_size = plane.width * plane.height * plane.depth * plane.type_size;
        entry_size += plane_size;

        if (plane.type_size != 1 && plane.type_size != 4 && plane.type_size != 8) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "plane {} has unsupported type size of {} bytes (expected 1, 4, or 8)",
                i,
                plane.type_size
            ));
        }
    }

    if (_buffer->max_entry_size_bytes < entry_size) [[unlikely]] {
        throw std::runtime_error(fmt::format(
            "cannot write {} bytes to buffer with maximum size of {} bytes",
            entry_size,
            _buffer->max_entry_size_bytes
        ));
    }

    std::size_t idx = (_buffer->uid + 1) % BUFFER_CNT;
    std::uint64_t x = _buffer->metadata[idx].v_a + 1;

    // BEGIN CRTITICAL SECTION ========
    // First, copy all data to ensure it's fully written before metadata is visible
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < plane_count; i++) {
        const auto& plane = planes[i];
        const std::size_t plane_size = plane.width * plane.height * plane.depth * plane.type_size;

        std::memcpy(
            _buffer->data + (idx * _buffer->max_entry_size_bytes) + cursor,
            plane.data,
            plane_size
        );
        cursor += plane_size;
    }

    // Memory barrier to ensure all data writes are visible before metadata updates
    std::atomic_thread_fence(std::memory_order_release);

    // Now set metadata after data is fully written
    _buffer->metadata[idx].v_a = x;
    _buffer->metadata[idx].acquisition_time = acquisition_time;
    _buffer->metadata[idx].total_size = entry_size;
    _buffer->metadata[idx].plane_count = plane_count;

    cursor = 0;
    for (std::size_t i = 0; i < plane_count; i++) {
        const auto& plane = planes[i];
        PlaneMetadata plane_meta {};
        plane_meta.width = plane.width;
        plane_meta.height = plane.height;
        plane_meta.depth = plane.depth;
        plane_meta.type_size = plane.type_size;
        plane_meta.offset = cursor;
        // copy name (optional)
        if (plane.name != nullptr) {
            std::strncpy(plane_meta.name, plane.name, PLANE_NAME_MAX_LEN - 1);
            plane_meta.name[PLANE_NAME_MAX_LEN - 1] = '\0';
        } else {
            plane_meta.name[0] = '\0';
        }

        _buffer->metadata[idx].planes[i] = plane_meta;

        const std::size_t plane_size = plane.width * plane.height * plane.depth * plane.type_size;
        cursor += plane_size;
    }

    for (std::size_t i = plane_count; i < MAX_PLANE_CNT; i++) {
        PlaneMetadata pm { 0, 0, 0, 0, 0, { 0 } };
        pm.name[0] = '\0';
        _buffer->metadata[idx].planes[i] = pm;
    }

    _buffer->metadata[idx].width = planes[0].width;
    _buffer->metadata[idx].height = planes[0].height;
    _buffer->metadata[idx].depth = planes[0].depth;
    _buffer->metadata[idx].type_size = planes[0].type_size;

    // Set v_b last to make the frame visible to readers
    _buffer->metadata[idx].v_b = x;

    // END CRITICAL SECTION =========

    // allow read frame to read;
    _buffer->uid.fetch_add(1);
    pthread_cond_broadcast(&_buffer->cond);

    return SUCCESS;
}

int Block::read_frame(Frame& frame, bool block_thread) {
    if (_buffer->deleted) {
        return FRAMEWORK_DELETED;
    }

    int mutex_errno = pthread_mutex_lock(&_buffer->cond_mutex);
    if (mutex_errno == EOWNERDEAD) {
        pthread_mutex_consistent(&_buffer->cond_mutex);
        mutex_errno = pthread_mutex_lock(&_buffer->cond_mutex);

        if (mutex_errno != 0) {
            _buffer->deleted = true;
            throw std::runtime_error("Failed to lock mutex: " + std::string(strerror(mutex_errno)));
        }
    }

    if (block_thread && frame.uid >= _buffer->uid) {
        std::cout << "sleep now" << std::endl;
        struct timespec time_to_wait;
        struct timeval now;
        gettimeofday(&now, NULL);

        time_to_wait.tv_sec = now.tv_sec + 1;
        time_to_wait.tv_nsec = (now.tv_usec) * 1000UL;

        if (time_to_wait.tv_nsec >= 1000000000) {
            time_to_wait.tv_sec += 1;
            time_to_wait.tv_nsec -= 1000000000;
        }

        pthread_cond_timedwait(&_buffer->cond, &_buffer->cond_mutex, &time_to_wait);
    }
    pthread_mutex_unlock(&_buffer->cond_mutex);

    if (frame.uid >= _buffer->uid.load()) {
        return NO_NEW_FRAME;
    }

    if (frame.size() < _buffer->max_entry_size_bytes) {
        frame.data = std::realloc(frame.data, _buffer->max_entry_size_bytes);
    }

    std::uint64_t v_a, v_b;
    do {
        std::size_t idx = _buffer->uid % BUFFER_CNT;
        v_b = _buffer->metadata[idx].v_b.load();
        frame.width = _buffer->metadata[idx].width;
        frame.height = _buffer->metadata[idx].height;
        frame.depth = _buffer->metadata[idx].depth;
        frame.type_size = _buffer->metadata[idx].type_size;
        frame.acquisition_time = _buffer->metadata[idx].acquisition_time;
        frame.uid = _buffer->uid.load();
        frame.total_size = _buffer->metadata[idx].total_size;
        frame.plane_count = static_cast<std::size_t>(_buffer->metadata[idx].plane_count);

        for (std::size_t i = 0; i < MAX_PLANE_CNT; i++) {
            const PlaneMetadata& plane_meta = _buffer->metadata[idx].planes[i];
            frame.planes[i].width = static_cast<std::size_t>(plane_meta.width);
            frame.planes[i].height = static_cast<std::size_t>(plane_meta.height);
            frame.planes[i].depth = static_cast<std::size_t>(plane_meta.depth);
            frame.planes[i].type_size = static_cast<std::size_t>(plane_meta.type_size);
            frame.planes[i].offset = static_cast<std::size_t>(plane_meta.offset);
            std::strncpy(frame.planes[i].name, plane_meta.name, PLANE_NAME_MAX_LEN - 1);
            frame.planes[i].name[PLANE_NAME_MAX_LEN - 1] = '\0';
        }

        std::memcpy(
            frame.data,
            _buffer->data + (idx * _buffer->max_entry_size_bytes),
            frame.size()
        );
        v_a = _buffer->metadata[idx].v_a.load();
        // std::cout << "repeat" << std::endl;
    } while (v_a != v_b);

    return SUCCESS;
}

const std::size_t Block::shm_size() const noexcept {
    return sizeof(Buffer) + _buffer->max_entry_size_bytes * BUFFER_CNT;
}

const std::size_t Block::max_buffer_size() const noexcept {
    return _buffer->max_entry_size_bytes;
}

///////////////////////////////////////////////////////////////////////////////
/// Buffer
///////////////////////////////////////////////////////////////////////////////
Frame::Frame() noexcept:
    width(0),
    height(0),
    depth(0),
    type_size(0),
    acquisition_time(0),
    uid(0),
    total_size(0),
    plane_count(0) {
    data = std::malloc(64);
    for (std::size_t i = 0; i < MAX_PLANE_CNT; i++) {
        planes[i] = FramePlane {};
    }
}

Frame::~Frame() noexcept {
    if (data != nullptr) {
        free(data);
    }
}

} // namespace cmf
