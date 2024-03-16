#pragma once

/**
 * @file app/base.h
 * @author sailing-innocent
 * @date 2023-12-26
 * @brief the Base Inno App
*/

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include "SailInno/core/runtime.h"

namespace sail::inno::app {

class SAIL_INNO_API BaseApp {
public:
	BaseApp() = default;
	virtual ~BaseApp() = default;
	virtual void create(luisa::string& cwd, luisa::string& device_name) = 0;

protected:
	S<luisa::compute::Device> mp_device;
	S<luisa::compute::Stream> mp_stream;
};

}// namespace sail::inno::app