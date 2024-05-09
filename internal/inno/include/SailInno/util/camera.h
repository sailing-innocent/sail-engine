#pragma once
/**
 * @file util/camera.h
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief Luisa Compute Based Camera
 */

#include "SailInno/config.h"
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

namespace sail::inno {

struct CameraData {
	luisa::compute::float3 cx;
	luisa::compute::float3 cy;
	luisa::compute::float3 cz;
	luisa::compute::float3 pos;
	float fov_rad;	   // in rad
	float aspect_ratio;// w / h
};

}// namespace sail::inno

// clang-format off

LUISA_STRUCT(sail::inno::CameraData, cx, cy, cz, pos, fov_rad, aspect_ratio) {
    [[nodiscard]] auto generate_ray(luisa::compute::Expr<luisa::compute::float2> p /*normalized coordinate*/) const noexcept {
        // flip-z coordinate system
        auto wi_local = make_float3(p * tan(0.5f * fov_rad), -1.0f);
        auto wi_world = normalize(aspect_ratio * wi_local.x * cx + wi_local.y * cy + wi_local.z * cz);
        return make_ray(pos, wi_world); 
    }
};

// clang-format on

namespace sail::inno {

class SAIL_INNO_API Camera {
	using float3 = luisa::float3;
	using float4 = luisa::float4;
	using float4x4 = luisa::float4x4;

public:
	enum class ProjType {
		Perspective,
		Orthographic
	};
	enum class CoordType {
		FlipZ,
		FlipY
	};

private:
	CameraData m_data;
	ProjType m_proj_type = ProjType::Perspective;
	CoordType m_coord_type = CoordType::FlipZ;

public:
	Camera(
		float3 p_from = luisa::make_float3(0.0f, -1.0f, 0.0f),
		float3 p_target = luisa::make_float3(0.0f, 0.0f, 0.0f),
		float3 p_up = luisa::make_float3(0.0f, 0.0f, 1.0f),
		CoordType coord_type = CoordType::FlipZ,
		float aspect = 1.0f, float fov_rad = luisa::pi_over_two) noexcept;

	// getter
	float fov_rad() const noexcept { return m_data.fov_rad; }
	float fov_deg() const noexcept { return m_data.fov_rad * 180.0f / luisa::pi; }
	float aspect_ratio() const noexcept { return m_data.aspect_ratio; }
	float3 pos() const noexcept { return m_data.pos; }
	float3 cx() const noexcept { return m_data.cx; }
	float3 cy() const noexcept { return m_data.cy; }
	float3 cz() const noexcept { return m_data.cz; }
	float4 camera_primitive(int width, int height) const noexcept;
	float4 camera_primitive() const noexcept;
	ProjType proj_type() const noexcept { return m_proj_type; }
	CoordType coord_type() const noexcept { return m_coord_type; }

	// set data
	void set_aspect_ratio(float aspect_ratio) noexcept { m_data.aspect_ratio = aspect_ratio; }
	void set_fov_rad(float fov_rad) noexcept { m_data.fov_rad = fov_rad; }
	void set_near(float _near) noexcept { m_near = _near; }
	void set_far(float _far) noexcept { m_far = _far; }
	// set external matrix
	float4x4 _proj_matrix;
	float4x4 _view_matrix;
	bool _external_matrix = false;// use external view matrix and projection matrix

	float m_near = 0.1f;
	float m_far = 100.0f;

public:
	// transforms
	float4x4 view_matrix() noexcept;
	float4x4 proj_matrix() noexcept;
};

}// namespace sail::inno