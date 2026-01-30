#include <string>
#pragma once
namespace filelock {

/**
 * @class Filelock
 * @brief A class implementing a file-based locking mechanism.
 *
 * The Filelock class is responsible for applying an exclusive lock using the file locking mechanisms 
 * provided by the operating system. This ensures multi-process critical sections. Copying or assigning Filelock
 * objects is prohibited to prevent unintentional multiple locks on the same file.
 */
class Filelock {
private:
    /// @brief The path to the lock file used for synchronization.
	std::string _filepath;

    /// @brief File descriptor associated with the lock file.
	int _fd;

public:
    /**
     * @brief Constructs a new `Filelock` object and acquires the file lock.
     *
     * This constructor attempts to lock the file specified by the `filepath`.
     * If the file is already locked by another process, it may block or fail,
     * depending on the implementation details.
     *
     * @param filepath The path to the lock file used for synchronization.
     */
	explicit Filelock(const std::string& filepath);

    /**
     * @brief Deleted copy constructor to prevent copying of `Filelock` objects.
     *
     * Copying a `Filelock` object is disallowed to prevent multiple instances
     * from managing the same lock, which could lead to inconsistent states.
     */
	Filelock(const Filelock&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent assignment of `Filelock` objects.
     *
     * Assignment is disallowed for the same reasons as copying; it prevents multiple
     * instances from interfering with the lock's management.
     */
	Filelock& operator=(const Filelock&) = delete;

    /**
     * @brief Destructor that unlocks the file and releases system resources.
     * 
     * When the Filelock object is destroyed, the file lock is automatically released. 
     * This ensures that the file becomes available to other processes once the lock is no longer needed.
     * The destructor is marked noexcept to guarantee that no exceptions are thrown during destruction.
     */
	~Filelock() noexcept;
};
} // namespace filelock