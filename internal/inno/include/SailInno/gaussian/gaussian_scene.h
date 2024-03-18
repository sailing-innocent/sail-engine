#pragma once

/**
 * @file util/scene/gpu_scene/gaussians.h
 * @author sailing-innocent
 * @date 2024-01-02
 * @brief The GPU Gaussian Scene
*/

#include "SailInno/util/scene/gpu_scene.h"
#include <luisa/runtime/buffer.h>
#include "sailInno/core/runtime.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno {

struct GaussiansData {
	int num_gaussians;
	Buffer<float> xyz_buf;
	Buffer<float> feat_buf;
	Buffer<float> opacity_buf;
	Buffer<float> scale_buf;
	Buffer<float> rot_buf;
};

}// namespace sail::inno

namespace sail::inno {

class SAIL_INNO_API GaussiansScene : public GPUScene {
protected:
	GaussiansData m_data;

public:
	GaussiansScene(const int num_gaussians = 0) noexcept;
	virtual ~GaussiansScene() = default;
	// life cycle
	void create(Device& device) noexcept override;
	void init(CommandList& cmdlist) noexcept override;
	// getter
	int num_gaussians() const noexcept { return m_data.num_gaussians; }
	BufferView<float> xyz() noexcept { return m_data.xyz_buf.view(); }
	BufferView<float> feat() noexcept { return m_data.feat_buf.view(); }
	BufferView<float> opacity() noexcept { return m_data.opacity_buf.view(); }
	BufferView<float> scale() noexcept { return m_data.scale_buf.view(); }
	BufferView<float> rot() noexcept { return m_data.rot_buf.view(); }

protected:
	// save for hosts
	std::vector<float> h_xyz;
	std::vector<float> h_feat;
	std::vector<float> h_opacity;
	std::vector<float> h_scale;
	std::vector<float> h_rot;
};

class SAIL_INNO_API SphereGaussians : public GaussiansScene {
public:
	SphereGaussians(int n_lon, int n_lat, float radius) noexcept;
	void init(CommandList& cmdlist) noexcept override;
	void set_feat(CommandList& cmdlist, luisa::float3 feat) noexcept;

private:
	int m_n_lat, m_n_lon;
	float m_radius;
};

}// namespace sail::inno