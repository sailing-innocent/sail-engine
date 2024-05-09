#pragma once

/**
 * @file util/scene/base.h
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief The GPU Scene Base Class
*/
#include "SailInno/config.h"
#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

namespace sail::inno {

class SAIL_INNO_API GPUScene {
public:
	GPUScene() = default;
	virtual ~GPUScene() = default;
	// life cycle
	virtual void create(luisa::compute::Device& device) noexcept = 0;
	virtual void init(luisa::compute::CommandList& cmdlist) noexcept = 0;// init scene data
};

}// namespace sail::inno