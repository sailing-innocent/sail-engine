/**
 * @file util/scene/gpu_scene/gaussians.h
 * @author sailing-innocent
 * @date 2024-01-02
 * @brief The GPU Gaussian Scene impl
*/
#include "SailInno/gaussian/gaussian_scene.h"
#include <random>

namespace sail::inno {

GaussiansScene::GaussiansScene(const int num_gaussians) noexcept {
	m_data.num_gaussians = num_gaussians;
}

void GaussiansScene::create(Device& device) noexcept {
	// create buffer
	m_data.xyz_buf = device.create_buffer<float>(m_data.num_gaussians * 3);
	m_data.feat_buf = device.create_buffer<float>(m_data.num_gaussians * 3);
	m_data.opacity_buf = device.create_buffer<float>(m_data.num_gaussians);
	m_data.scale_buf = device.create_buffer<float>(m_data.num_gaussians * 3);
	m_data.rot_buf = device.create_buffer<float>(m_data.num_gaussians * 4);
	h_xyz.resize(m_data.num_gaussians * 3);
	h_feat.resize(m_data.num_gaussians * 3);
	h_opacity.resize(m_data.num_gaussians);
	h_scale.resize(m_data.num_gaussians * 3);
	h_rot.resize(m_data.num_gaussians * 4);
}

void GaussiansScene::init(CommandList& cmdlist) noexcept {
	// randomly init
	std::default_random_engine random{std::random_device{}()};
	std::uniform_real_distribution<float> uniform;
	// init from 0-1
	for (auto i = 0; i < m_data.num_gaussians; i++) {
		h_xyz[3 * i + 0] = uniform(random);
		h_xyz[3 * i + 1] = uniform(random);
		h_xyz[3 * i + 2] = uniform(random);
		h_feat[3 * i + 0] = h_xyz[3 * i + 0];
		h_feat[3 * i + 1] = h_xyz[3 * i + 1];
		h_feat[3 * i + 2] = h_xyz[3 * i + 2];
		h_opacity[i] = 1.0f;
		h_scale[3 * i + 0] = 1.0f;
		h_scale[3 * i + 1] = 1.0f;
		h_scale[3 * i + 2] = 1.0f;
		// rxyz
		h_rot[4 * i + 0] = 1.0f;
		h_rot[4 * i + 1] = 0.0f;
		h_rot[4 * i + 2] = 0.0f;
		h_rot[4 * i + 3] = 0.0f;
	}
	// copy to device
	// LUISA_INFO("copy to device");
	cmdlist << m_data.xyz_buf.copy_from(h_xyz.data())
			<< m_data.feat_buf.copy_from(h_feat.data())
			<< m_data.opacity_buf.copy_from(h_opacity.data())
			<< m_data.scale_buf.copy_from(h_scale.data())
			<< m_data.rot_buf.copy_from(h_rot.data());
}

}// namespace sail::inno