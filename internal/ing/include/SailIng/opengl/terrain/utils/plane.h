#pragma once
#include <glad/gl.h>
#include <glm/glm.hpp>

namespace ing::terrain {

void initialize_plane_VAO(
	const int res,
	const int width,
	GLuint* plane_VAO,
	GLuint* plane_VBO,
	GLuint* plane_EBO);

}// namespace ing::terrain
