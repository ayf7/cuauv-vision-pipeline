// If there is a problem, contact Jeffrey Qian on Slack. Requires [Jeffrey Qian]
// still exists.
#pragma once

#include <string>

namespace cmf {
/// @brief number of frames to store in each buffer
inline constexpr std::size_t BUFFER_CNT = 3;

/// @brief maximum number of planes supported in a single frame
inline constexpr std::size_t MAX_PLANE_CNT = 4;

/// @brief maximum number of bytes reserved for a plane name (including NUL)
inline constexpr std::size_t PLANE_NAME_MAX_LEN = 32;

/// @brief read success
inline constexpr int SUCCESS = 0;

/// @brief no new frame to read
inline constexpr int NO_NEW_FRAME = 1;

/// @brief buffer is marked for deletion and should not be read from
inline constexpr int FRAMEWORK_DELETED = 2;

/// @brief File stub for page mappings
inline const std::string BLOCK_STUB { "/dev/shm/auv_visiond_" };

/// @brief Lock file
inline const std::string GLOBAL_LOCK { "/dev/shm/auv_visiond.lock" };

struct Buffer;
struct FrameMetadata;

/**
 * @struct Frame
 * @brief Represents an image with metadata like dimensions, type, and
 * acquisition time.
 */
struct FramePlane {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t depth = 0;
    std::size_t type_size = 0;
    std::size_t offset = 0;
    char name[PLANE_NAME_MAX_LEN] {};
};

struct Frame {
    /// @brief Width of the frame.
    std::size_t width = 0;

    /// @brief Height of the frame.
    std::size_t height = 0;

    /// @brief Depth of the frame.
    std::size_t depth = 0;

    /// @brief Size of each element type in bytes.
    std::size_t type_size = 0;

    /// @brief Timestamp representing the acquisition time of the frame.
    std::uint64_t acquisition_time = 0;

    /// @brief Unique identifier for the frame, like a version number.
    std::uint64_t uid = 0;

    /// @brief Pointer to the raw data of the frame.
    void* data = nullptr;

    /// @brief Total size of the serialized frame in bytes.
    std::size_t total_size = 0;

    /// @brief Number of planes stored in the frame.
    std::size_t plane_count = 0;

    /// @brief Per-plane metadata describing shapes and layout within the buffer.
    FramePlane planes[MAX_PLANE_CNT] {};

    Frame() noexcept;
    ~Frame() noexcept;

    /// @brief Calculates the total size of the frame's data.
    inline std::size_t size() const {
        return total_size;
    }
};

/**
 * @struct FramePlaneWrite
 * @brief Describes a plane to be written into a block.
 */
struct FramePlaneWrite {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t depth = 0;
    std::size_t type_size = 0;
    const void* data = nullptr;
    const char* name = nullptr;
};

/**
 * @class Block
 * @brief A volatile memory-backed object capable of being shared between
 * multiple processes.
 */
class Block {
public:
    /// @brief  A block accessing nothing is ill-defined
    Block() = delete;

    /**
     * @brief Construct a new Block object and reserves the volatile-memory
     *
     * @param direction the buffer name that should be created
     * @param max_entry_size_bytes the number of bytes reserved for a single entry
     * in the buffer. If the buffer name already exists, `max_entry_size` should
     * be the same as the value found in the buffer.
     */
    Block(const std::string& direction, const std::size_t max_entry_size_bytes);

    /**
     * @brief Open a block object if it exists. Else, throws a `filesystem_error`
     *
     * @param direction buffer name to open
     */
    Block(const std::string& direction);

    /// @brief copying makes it difficult to safely reason about behavior
    Block(const Block&) = delete;

    /// @brief copying makes it difficult to safely reason about behavior
    Block& operator=(const Block&) = delete;

    /// @brief move constructor that takes an R value
    Block(Block&& other) noexcept;

    /// @brief move assignment that takes an R value
    Block& operator=(Block&& other);

    ~Block();

    /**
     * @brief write data in a raw pointer to the block
     *
     * @param acquisition_time time when frame was acquired in milliseconds
     * @param width width of image
     * @param height height of image
     * @param depth depth of image
     * @param type_size datatype width of image
     * @param data pointer to the image that is width*height*depth*type_size bytes
     * long.
     */
    int write_frame(
        std::uint64_t acquisition_time,
        std::size_t width,
        std::size_t height,
        std::size_t depth,
        std::size_t type_size,
        const void* data
    ) const noexcept;

    /**
     * @brief write multiple image planes to the block
     *
     * @param acquisition_time time when frame was acquired in milliseconds
     * @param planes pointer to an array of plane descriptors
     * @param plane_count number of planes in the array
     */
    int write_frame(
        std::uint64_t acquisition_time,
        const FramePlaneWrite* planes,
        std::size_t plane_count
    ) const noexcept;

    /**
     * @brief read frame from block, using the previous frame to determine if
     * there is a newer frame
     *
     * @param frame contains the previous frame to overwrite
     * @param block_thread if true, read_frame blocks for a few seconds to grab a
     * new frame as soon as it's ready
     * @return int read return code
     */
    int read_frame(Frame& frame, bool block_thread);

    /// @brief get the underlying file that backs the buffer
    inline const std::string& filename() const noexcept {
        return _filename;
    }

    /// @brief get direction name
    inline const std::string& direction() const noexcept {
        return _direction;
    }

    /// @brief get the size in bytes of the buffer
    const std::size_t shm_size() const noexcept;

    /// @brief get the maximum buffer size
    const std::size_t max_buffer_size() const noexcept;

    inline bool is_valid() {
        return _buffer != nullptr;
    }

private:
    void close_block();

private:
    std::string _filename = "";
    std::string _direction = "";
    bool _creator;
    Buffer* _buffer;
};

} // namespace cmf
