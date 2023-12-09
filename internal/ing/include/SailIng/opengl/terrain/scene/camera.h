#pragma once
/** 
 * @file terrain/camera.h
 * @author sailing-innocent
 * @date 2023-08-30
 * @brief the terrain camera
 */
#include "SailBase/config.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace ing::terrain {

enum CAMERA_MOVEMENT {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

// Default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2000.f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 60.0f;
const float MAX_FOV = 100.0f;

// An abstract camera class that processes input and calculates the
// corresponding Euler Angles, Vectors and Matrices for use in OpenGL

class SAIL_ING_API Camera {
public:
	glm::vec3 m_position;
	glm::vec3 m_front;
	glm::vec3 m_up;
	glm::vec3 right;
	glm::vec3 m_world_up;
	// Euler Angles
	float m_yaw;
	float m_pitch;
	// Camera Options
	float movement_speed;
	float mouse_sensitivity;
	float zoom;

	// Constructor with vectors
	Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
		   glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
		   float yaw = YAW,
		   float pitch = PITCH)
		: m_front(glm::vec3(0.0f, 0.0f, -5.0f)), movement_speed(SPEED), mouse_sensitivity(SENSITIVITY), zoom(ZOOM) {
		m_position = position;
		m_world_up = up;
		m_yaw = yaw;
		m_pitch = pitch;
		update_camera_vectors();
	};

	// construct with scalar vectors

	glm::mat4 get_view_matrix() {
		update_camera_vectors();
		return glm::lookAt(m_position, m_position + m_front, m_up);
	}

private:
	void update_camera_vectors() {
		glm::vec3 front;
		front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
		front.y = sin(glm::radians(m_pitch));
		front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
		m_front = glm::normalize(front);
		right = glm::normalize(glm::cross(m_front, m_world_up));
		m_up = glm::normalize(glm::cross(right, m_front));
	}
};

}// namespace ing::terrain