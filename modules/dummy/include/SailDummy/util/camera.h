#pragma once
/**
 * @file camera.h
 * @brief Dummy Camera
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailDummy/config.h"
#include <glm/ext.hpp>
#include <glm/glm.hpp>

namespace sail::dummy {

class SAIL_DUMMY_API Camera {
public:
	enum struct CameraType : unsigned int {
		kFlipZ,
		kFlipY,
	};
	explicit Camera(const glm::vec3& pos,
					const glm::vec3& target,
					const glm::vec3& up,
					const float fov_deg,
					const float aspect,
					const float znear,
					const float zfar,
					CameraType type = CameraType::kFlipZ);
	const glm::mat4& view_matrix() noexcept;
	const glm::mat4& proj_matrix() noexcept;

	// getter & setter
	glm::vec3 cam_pos() const noexcept { return m_cam_pos; }
	void set_cam_pos(const glm::vec3& pos) noexcept {
		m_cam_pos = pos;
		m_is_view_dirty = true;
	}
	glm::vec3 right() const noexcept { return m_right; }
	void set_right(const glm::vec3& right) noexcept {
		m_right = right;
		m_is_view_dirty = true;
	}
	glm::vec3 dir() const noexcept { return m_dir; }
	void set_dir(const glm::vec3& dir) noexcept {
		m_dir = dir;
		m_is_view_dirty = true;
	}
	glm::vec3 up() const noexcept { return m_up; }
	void set_up(const glm::vec3& up) noexcept {
		m_up = up;
		m_is_view_dirty = true;
	}
	void set_view_matrix(const glm::mat4& view_matrix) noexcept {
		m_view_matrix = view_matrix;
		m_is_view_dirty = false;
	}
	void set_proj_matrix(const glm::mat4& proj_matrix) noexcept {
		m_proj_matrix = proj_matrix;
		m_is_proj_dirty = false;
	}
	float fov() const noexcept { return m_fov_rad; }
	void set_fov(const float fov_rad) noexcept {
		m_fov_rad = fov_rad;
		m_is_proj_dirty = true;
	}
	float aspect() const noexcept { return m_aspect; }
	void set_aspect(const float aspect) noexcept {
		m_aspect = aspect;
		m_is_proj_dirty = true;
	}
	float znear() const noexcept { return m_znear; }
	void set_znear(const float znear) noexcept {
		m_znear = znear;
		m_is_proj_dirty = true;
	}
	float zfar() const noexcept { return m_zfar; }
	void set_zfar(const float zfar) noexcept {
		m_zfar = zfar;
		m_is_proj_dirty = true;
	}
	CameraType type() const noexcept { return m_type; }
	void set_type(CameraType type) noexcept { m_type = type; }

private:
	glm::vec3 m_cam_pos;
	glm::vec3 m_right;
	glm::vec3 m_dir;
	glm::vec3 m_up;
	glm::mat4 m_view_matrix;
	glm::mat4 m_proj_matrix;
	bool m_is_view_dirty = true;
	bool m_is_proj_dirty = true;
	float m_fov_rad;
	float m_aspect;
	float m_znear;
	float m_zfar;
	CameraType m_type;
};

}// namespace sail::dummy