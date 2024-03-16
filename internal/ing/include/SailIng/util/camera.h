#pragma once
/**
 * @file util/camera.h
 * @brief The ING Camera Utils
 * @date 2023-09-29
 * @author sailing-innocent
 */
#include "SailBase/config.h"
#include <glm/ext.hpp>// perspective, translate, rotate, scale
#include <glm/glm.hpp>// vec2

namespace sail::ing {

struct SAIL_ING_API INGFlipZCamera {
	glm::vec3 m_cam_pos;
	glm::vec3 m_right;
	glm::vec3 m_dir;
	glm::vec3 m_up;
	glm::mat4 m_view_matrix;
	glm::mat4 m_proj_matrix;
	float m_tan_half_fov;
	float m_aspect;
	float m_znear;
	float m_zfar;
	explicit INGFlipZCamera(const glm::vec3& pos,
							const glm::vec3& target,
							const glm::vec3& up,
							const float fov,
							const float aspect,
							const float znear,
							const float zfar);
};

struct SAIL_ING_API INGFlipYCamera {
	glm::vec3 m_cam_pos;
	glm::vec3 m_x;
	glm::vec3 m_z;
	glm::vec3 m_y;
	glm::mat4 m_view_matrix;
	glm::mat4 m_proj_matrix;
	float m_tan_half_fov;
	float m_aspect;
	float m_znear;
	float m_zfar;
	explicit INGFlipYCamera(const glm::vec3& pos,
							const glm::vec3& target,
							const glm::vec3& up,
							const float fov,
							const float aspect,
							const float znear,
							const float zfar);
};

}// namespace sail::ing