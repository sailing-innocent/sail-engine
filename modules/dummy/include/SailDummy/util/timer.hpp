#pragma once
/**
 * @file timer.hpp
 * @brief Sail Timer Class (headeronly)
 * @date 2023-12-15
 * @author sailing-innocent
*/
#include <chrono>

namespace sail::dummy {

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

// a timer<Precision> class
template<typename T>
class Timer {
public:
	Timer() { reset(); }// reset
	// default destructor
	// get
	T total_time() const {
		if (m_stopped)
			return static_cast<T>(std::chrono::duration_cast<std::chrono::duration<T>>(
									  m_stop_time - m_base_time)
									  .count()) -
				   m_paused_time;
		else
			return static_cast<T>(std::chrono::duration_cast<std::chrono::duration<T>>(
									  m_curr_time - m_base_time)
									  .count()) -
				   m_paused_time;
	}											 // in s
	T delta_time() const { return m_delta_time; }// in s
	T toc() const { return m_delta_time; }		 // another name for delta_time
	// method
	void start() {
		TimePoint now = std::chrono::high_resolution_clock::now();
		m_prev_time = now;
		if (m_stopped) {
			m_paused_time =
				m_paused_time + std::chrono::duration_cast<std::chrono::duration<T>>(now - m_stop_time)
									.count();
			m_stopped = false;
		}
	}
	void stop() {
		if (!m_stopped) {
			TimePoint now = std::chrono::high_resolution_clock::now();
			m_stop_time = now;
			m_stopped = true;
		}
	}

	void reset() {
		TimePoint now = std::chrono::high_resolution_clock::now();
		m_delta_time = -1.0;
		m_stopped = false;
		m_base_time = now;
		m_paused_time = static_cast<T>(0.0);
		m_prev_time = now;
		m_curr_time = now;
	}

	void tick() {
		if (m_stopped) {
			m_delta_time = 0.0;
			return;
		}
		TimePoint now = std::chrono::high_resolution_clock::now();
		m_curr_time = now;
		m_delta_time =
			std::chrono::duration_cast<std::chrono::duration<T>>(m_curr_time - m_prev_time)
				.count();
		if (m_delta_time < 0.0)
			m_delta_time = static_cast<T>(0.0);
	}
	T elapsed() {
		return std::chrono::duration_cast<std::chrono::duration<T>>(m_stop_time - m_prev_time)
			.count();
	}
	int elapsed_ms() {
		return std::chrono::duration_cast<std::chrono::milliseconds>(m_stop_time - m_prev_time)
			.count();
	}// in ms
	int elapsed_us() {
		return std::chrono::duration_cast<std::chrono::microseconds>(m_stop_time - m_prev_time)
			.count();
	}// in us

private:
	TimePoint m_base_time;
	TimePoint m_stop_time;
	TimePoint m_prev_time;
	TimePoint m_curr_time;
	bool m_stopped;
	T m_paused_time;
	T m_delta_time;
};

}// namespace sail::dummy