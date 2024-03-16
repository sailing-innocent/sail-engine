/**
 * @file gaussian/sphere_scene.cpp
 * @author sailing-innocent
 * @date 2024-01-03
 * @brief The GPU Gaussian Sphere Scene impl
*/

#include "SailInno/gaussian/gaussian_scene.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno {

SphereGaussians::SphereGaussians(int n_lon, int n_lat, float radius) noexcept {
	m_data.num_gaussians = n_lon * n_lat;
	m_n_lon = n_lon;
	m_n_lat = n_lat;
	m_radius = radius;
}

void SphereGaussians::init(CommandList& cmdlist) noexcept {
	auto eps = 1e-3;
	for (auto i = 0; i < m_n_lat; i++) {
		for (auto j = 0; j < m_n_lon; j++) {
			auto idx = i * m_n_lon + j;
			auto theta = i * 2.0f * luisa::pi / m_n_lat;
			auto phi = j * (luisa::pi - 2 * eps) / m_n_lon - luisa::pi / 2.0f + eps;
			h_xyz[3 * idx + 0] = m_radius * std::cos(theta) * std::cos(phi);
			h_xyz[3 * idx + 1] = m_radius * std::sin(theta) * std::cos(phi);
			h_xyz[3 * idx + 2] = m_radius * std::sin(phi);
			h_feat[3 * idx + 0] = h_xyz[3 * idx + 0] / 2.0f / m_radius + 0.5f;
			h_feat[3 * idx + 1] = h_xyz[3 * idx + 1] / 2.0f / m_radius + 0.5f;
			h_feat[3 * idx + 2] = h_xyz[3 * idx + 2] / 2.0f / m_radius + 0.5f;

			h_opacity[idx] = 1.0f;
			h_scale[3 * idx + 0] = 1.0f * 0.1f;
			h_scale[3 * idx + 1] = 1.0f * 0.1f;
			h_scale[3 * idx + 2] = 1.0f * 0.1f;
			// rxyz
			h_rot[4 * idx + 0] = 1.0f;
			h_rot[4 * idx + 1] = 0.0f;
			h_rot[4 * idx + 2] = 0.0f;
			h_rot[4 * idx + 3] = 0.0f;
		}
	}
	// copy to device
	// LUISA_INFO("copy to device");
	cmdlist << m_data.xyz_buf.copy_from(h_xyz.data())
			<< m_data.feat_buf.copy_from(h_feat.data())
			<< m_data.opacity_buf.copy_from(h_opacity.data())
			<< m_data.scale_buf.copy_from(h_scale.data())
			<< m_data.rot_buf.copy_from(h_rot.data());
}

void SphereGaussians::set_feat(CommandList& cmdlist, luisa::float3 feat) noexcept {
	for (auto i = 0; i < m_data.num_gaussians; i++) {
		h_feat[3 * i + 0] = feat.x;
		h_feat[3 * i + 1] = feat.y;
		h_feat[3 * i + 2] = feat.z;
	}
	cmdlist << m_data.feat_buf.copy_from(h_feat.data());
}

}// namespace sail::inno