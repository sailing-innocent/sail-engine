/**
 * @file camera.cpp
 * @brief The Implementation of Dummy Camera
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailDummy/util/camera.h"

namespace sail::dummy {

Camera::Camera(const glm::vec3& pos,
			   const glm::vec3& target,
			   const glm::vec3& up,
			   const float fov_deg,
			   const float aspect,
			   const float znear,
			   const float zfar,
			   CameraType type) {
	m_dir = glm::normalize(target - pos);
	m_right = glm::normalize(glm::cross(m_dir, up));
	m_up = glm::cross(m_right, m_dir);

	m_cam_pos = pos;
	m_fov_rad = glm::radians(fov_deg);
	m_aspect = aspect;
	m_znear = znear;
	m_zfar = zfar;

	m_view_matrix = glm::lookAt(pos, target, up);
	m_proj_matrix = glm::perspective(m_fov_rad, aspect, znear, zfar);
}

const glm::mat4& Camera::view_matrix() noexcept {
	if (m_is_view_dirty) {
		m_view_matrix = glm::lookAt(m_cam_pos, m_cam_pos + m_dir, m_up);
		m_is_view_dirty = false;
	}
	return m_view_matrix;
}

const glm::mat4& Camera::proj_matrix() noexcept {
	if (m_is_proj_dirty) {
		m_proj_matrix = glm::perspective(m_fov_rad, m_aspect, m_znear, m_zfar);
		m_is_proj_dirty = false;
	}
	return m_proj_matrix;
}

}// namespace sail::dummy