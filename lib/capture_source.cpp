/**
 * @file flir.cpp
 * @author Jeffrey Qian
 * @brief 
 * @version 1.0
 * @date 2025-03-01
 */

#include "include/capture_source.hpp"
using namespace std::chrono;

namespace capture_source {
FpsLimiter::FpsLimiter(std::optional<uint32_t> fps)
	: _target_frame_time_ms(fps.has_value() ? 1000 / *fps : 0)
	, _last_frame_time(high_resolution_clock::now()) { }

uint64_t FpsLimiter::tick() {
	auto now = high_resolution_clock::now();
	uint64_t time_elapsed = duration_cast<milliseconds>(now - _last_frame_time).count();

	if(time_elapsed < _target_frame_time_ms) {
		std::this_thread::sleep_for(milliseconds(_target_frame_time_ms - time_elapsed));
	}

	_last_frame_time = high_resolution_clock::now();
	return duration_cast<milliseconds>(_last_frame_time.time_since_epoch()).count();
}

void CaptureSource::run_until_complete() {
	for(auto& t : _thread_handlers) {
		t.join();
	}
}

CaptureSource::CaptureSource()
	: _quit_flag(false) { }

CaptureSource::~CaptureSource() { }

}; // namespace capture_source