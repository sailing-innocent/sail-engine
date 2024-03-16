#pragma once

/**
 * @file util/scene/gpu_scene/points.h
 * @author sailing-innocent
 * @date 2023/12/27
 * @brief The GPU Scene Based on Point Primitive
*/

#include "gpu_scene.h"
#include <luisa/runtime/buffer.h>

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno {

struct PointsData {
	int num_points;
	Buffer<float> xyz_buf;
	Buffer<float> color_buf;
};

}// namespace sail::inno

namespace sail::inno {

class SAIL_INNO_API PointsScene : public GPUScene {
	PointsData m_data;

public:
	PointsScene(const int num_points = 0) noexcept;
	~PointsScene() = default;
	// life cycle
	void create(Device& device) noexcept override;
	void init(CommandList& cmdlist) noexcept override;

public:
	// getter
	int num_points() const noexcept { return m_data.num_points; }
	BufferView<float> xyz() noexcept { return m_data.xyz_buf.view(); }
	BufferView<float> color() noexcept { return m_data.color_buf.view(); }

private:
	// save for hosts
	std::vector<float> h_xyz;
	std::vector<float> h_color;
};

}// namespace sail::inno