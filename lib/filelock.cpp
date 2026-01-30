#include "include/filelock.hpp"

#include <fcntl.h>
#include <iostream>
#include <fmt/format.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

namespace filelock {

Filelock::Filelock(const std::string& filepath)
	: _filepath(filepath)
	, _fd(open(_filepath.c_str(), O_RDWR | O_CREAT, S_IRWXU)) {

	if(_fd == -1) {
		throw std::runtime_error(
			fmt::format("Could not initialized lock {}: {}", filepath, strerror(errno)));
	}

	if(flock(_fd, LOCK_EX) == -1) {
		throw std::runtime_error(
			fmt::format("Could not acquire lock {}: {}", filepath, strerror(errno)));
	}
}

Filelock::~Filelock() {
	if(_fd != -1) {
		flock(_fd, LOCK_UN);
		close(_fd);
	}
}

}; // namespace filelock