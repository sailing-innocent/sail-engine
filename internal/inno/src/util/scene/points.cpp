/**
 * @file util/scene/points.cpp
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief The GPU Scene Based on Point Primitive Implementation
*/

#include "SailInno/util/scene/points.h"
#include <random>

namespace sail::inno {

PointsScene::PointsScene(const int num_points) noexcept {
	m_data.num_points = num_points;
}

void PointsScene::create(Device& device) noexcept {
	// create buffer
	m_data.xyz_buf = device.create_buffer<float>(m_data.num_points * 3);
	m_data.color_buf = device.create_buffer<float>(m_data.num_points * 3);
	h_xyz.resize(m_data.num_points * 3);
	h_color.resize(m_data.num_points * 3);
}

void PointsScene::init(CommandList& cmdlist) noexcept {
	// randomly init
	std::default_random_engine random{std::random_device{}()};
	std::uniform_real_distribution<float> uniform;
	// init from 0-1
	for (auto i = 0; i < m_data.num_points; i++) {
		h_xyz[3 * i + 0] = uniform(random);
		h_xyz[3 * i + 1] = uniform(random);
		h_xyz[3 * i + 2] = uniform(random);
		h_color[3 * i + 0] = h_xyz[3 * i + 0];
		h_color[3 * i + 1] = h_xyz[3 * i + 1];
		h_color[3 * i + 2] = h_xyz[3 * i + 2];
	}
	// copy to device
	// LUISA_INFO("copy to device");
	cmdlist << m_data.xyz_buf.copy_from(h_xyz.data())
			<< m_data.color_buf.copy_from(h_color.data());
}

}// namespace sail::inno