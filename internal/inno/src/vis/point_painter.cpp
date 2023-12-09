/**
 * @file vis/point_painter.cpp
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief 3D Point Painter Impl
 */

#include "SailInno/vis/point_painter.h"
#include <luisa/core/logging.h>

using namespace luisa;
using namespace luisa::compute;

// API

namespace sail::inno {

void PointPainter::create(Device& device) noexcept {
	mp_point_render = luisa::make_shared<render::PointRender>();
	mp_point_render->compile(device, m_stride);
	compile(device);
}

void PointPainter::paint(CommandList& cmdlist, ImageView<float> out_img, int w, int h) noexcept {
	cmdlist << (*ms_clear)(out_img, w, h, m_clear_color).dispatch(w, h);

	mp_point_render->render(cmdlist, out_img, w, h, m_point_num, m_point_xyz, m_point_color, m_view_matrix, m_proj_matrix);
}

void PointPainter::update_point(int point_num, BufferView<float> point_xyz, BufferView<float> point_color) noexcept {
	LUISA_INFO("update point {} with {} xyz and {} color", point_num, point_xyz.size(), point_color.size());
	m_point_num = point_num;
	m_point_xyz = point_xyz;
	m_point_color = point_color;
}

}// namespace sail::inno
