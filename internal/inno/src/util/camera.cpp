/**
 * @file util/camera.cpp
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief Luisa Compute Based Camera Impl
 */
#include "SailInno/util/camera.h"
#include <luisa/dsl/sugar.h>

namespace sail::inno {

using namespace luisa;
using namespace luisa::compute;

Camera::Camera(float3 p_from, float3 p_target, float3 p_up, CoordType coord_type, float aspect, float fov_rad) noexcept {
	// LUISA_INFO("init camera with lookat");
	// init default camera
	m_coord_type = coord_type;
	m_data.pos = p_from;
	m_data.cy = normalize(p_up);
	m_data.cz = -normalize(p_target - p_from);
	m_data.cx = normalize(cross(m_data.cy, m_data.cz));
	m_data.cy = normalize(cross(m_data.cz, m_data.cx));
	if (m_coord_type == CoordType::FlipY) {
		m_data.cy = -m_data.cy;
	}
	if (m_coord_type == CoordType::FlipZ) {
		m_data.cz = -m_data.cz;
	}
	m_data.fov_rad = fov_rad;
	m_data.aspect_ratio = aspect;
}

float4 Camera::camera_primitive(int width, int height) const noexcept {
	float fwidth = (float)width;
	float fheight = (float)height;
	float tan_fov_y = tanf(this->fov_rad() * 0.5f);
	float aspect = fwidth / fheight;
	float tan_fov_x = tan_fov_y * aspect;
	float focal_x = fwidth / 2.0f / tan_fov_x;
	float focal_y = fheight / 2.0f / tan_fov_y;
	float4 camera_primitive = luisa::make_float4(focal_x, focal_y, tan_fov_x, tan_fov_y);
	return camera_primitive;
}

float4 Camera::camera_primitive() const noexcept {
	float fheight = 1.0f;
	float tan_fov_y = tanf(this->fov_rad() * 0.5f);
	float aspect = this->aspect_ratio();
	float tan_fov_x = tan_fov_y * aspect;
	float focal_y = fheight / 2.0f / tan_fov_y;
	float focal_x = focal_y;
	float4 camera_primitive = luisa::make_float4(focal_x, focal_y, tan_fov_x, tan_fov_y);
	return camera_primitive;
}

float4x4 Camera::view_matrix() noexcept {
	if (!_external_matrix) {
		float3 z = cz();
		float3 x = cx();
		float3 y = cy();
		float3 c = pos();
		float4 col_1 = make_float4(x.x, y.x, z.x, 0.0f);
		float4 col_2 = make_float4(x.y, y.y, z.y, 0.0f);
		float4 col_3 = make_float4(x.z, y.z, z.z, 0.0f);
		float4 col_4 = make_float4(-dot(c, x), -dot(c, y), -dot(c, z), 1.0f);
		_view_matrix = make_float4x4(col_1, col_2, col_3, col_4);
	}
	return _view_matrix;
}

float4x4 Camera::proj_matrix() noexcept {
	if (!_external_matrix) {
		auto t = static_cast<float>(coord_type() == CoordType::FlipZ ? 1 : -1) * tan(0.5f * fov_rad());

		float n = m_near;
		float f = m_far;
		float as = aspect_ratio();
		// projection
		float4 col_1 = make_float4(1.0f / (t * as), 0.0f, 0.0f, 0.0f);
		float4 col_2 = make_float4(0.0f, 1.0f / t, 0.0f, 0.0f);
		float4 col_3 = make_float4(0.0f, 0.0f, 1.0f * f / (f - n), 1.0f);
		float4 col_4 = make_float4(0.0f, 0.0f, -2.0f * n * f / (f - n), 0.0f);
		_proj_matrix = make_float4x4(col_1, col_2, col_3, col_4);
	}
	return _proj_matrix;
}

}// namespace sail::inno