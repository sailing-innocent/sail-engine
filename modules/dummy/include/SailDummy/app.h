#pragma once
/**
 * @file app.h
 * @author sailing-innocent
 * @date 2024-04-26
 * @brief Dummy App Inteface
*/

namespace sail {

class DummyApp {
public:
	DummyApp() = default;
	virtual ~DummyApp() {}
	// disable copy and move
	DummyApp(const DummyApp&) = delete;
	DummyApp& operator=(const DummyApp&) = delete;
	DummyApp(DummyApp&&) = delete;
	DummyApp& operator=(DummyApp&&) = delete;
	// virtual functions
	virtual void init() = 0;
	virtual bool tick(int count) = 0;
	virtual void terminate() = 0;
};

}// namespace sail