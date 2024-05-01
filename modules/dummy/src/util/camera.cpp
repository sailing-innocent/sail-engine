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
						const float fov,
						const float aspect,
						const float znear,
						const float zfar,
						CameraType type) {
}

glm::mat4 Camera::view_matrix() noexcept {
	if (m_is_view_dirty) {
		m_view_matrix = glm::lookAt(m_cam_pos, m_cam_pos + m_dir, m_up);
		m_is_view_dirty = false;
	}
	return m_view_matrix;
}

glm::mat4 Camera::proj_matrix() noexcept {
	if (m_is_proj_dirty) {
		m_proj_matrix = glm::perspective(m_fov, m_aspect, m_znear, m_zfar);
		m_is_proj_dirty = false;
	}
	return m_proj_matrix;
}

}// namespace sail::dummy