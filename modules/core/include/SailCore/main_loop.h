#pragma once
/**
 * @file main_loop.h
 * @author sailing-innocent
 * @brief MainLoop Node
 * @date 2024-03-16
 */
#include "SailBase/config.h"

namespace sail {

class SAIL_CORE_API MainLoop {
public:
	virtual void initialize();
	virtual void physics_process();
	virtual void process();
	virtual void finalize();
};

};// namespace sail