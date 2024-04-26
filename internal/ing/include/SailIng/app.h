#pragma once
/**
 * @file app/app.h
 * @author sailing-innocent
 * @date 2023-02-25
 * @brief the basic INGApp class
 */
#include "SailBase/config.h"

namespace sail::ing {

class INGApp {
public:
	INGApp() = default;
	virtual ~INGApp() {}
	virtual void init() = 0;
	virtual bool tick(int count) = 0;
	virtual void terminate() = 0;
};

}// namespace sail::ing
