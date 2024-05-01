/**
 * @file primitive.cpp
 * @brief The Implementation of GL Primitive App
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailGL/dummy_app/primitive.h"

namespace sail::gl {
using std::make_unique;

GLPrimitiveApp::GLPrimitiveApp() noexcept {
	mp_primitive_root = make_unique<GLPrimitive>();
	mp_triangles = make_unique<GLTriangleList>();
	mp_lines = make_unique<GLLineList>();
	mp_points = make_unique<GLPointList>();
}

void GLPrimitiveApp::init() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	init_window(m_resw, m_resh);
	init_buffers();
	init_shaders();
}

void GLPrimitiveApp::init_buffers() {
	m_offset_state.triangle_start = 0;
	mp_primitive_root->appendPrimitive(*mp_triangles);
	m_offset_state.triangle_end = mp_primitive_root->indicies().size();

	m_offset_state.line_start = m_offset_state.triangle_end;
	mp_primitive_root->appendPrimitive(*mp_lines);
	m_offset_state.line_end = mp_primitive_root->indicies().size();

	m_offset_state.point_start = m_offset_state.line_end;
	mp_primitive_root->appendPrimitive(*mp_points);
	m_offset_state.point_end = mp_primitive_root->indicies().size();

	// gen buffers
	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &EBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, mp_primitive_root->vertices().size() * sizeof(float), mp_primitive_root->vertices().data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mp_primitive_root->indicies().size() * sizeof(unsigned int), mp_primitive_root->indicies().data(), GL_STATIC_DRAW);

	// aPos
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// aColor
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(4 * sizeof(float)));
	glEnableVertexAttribArray(1);
}

void GLPrimitiveApp::init_shaders() {
	mp_shader_program = make_unique<ShaderProgram>("scene");
	mp_shader_program
		->attach_shader(ShaderBase("assets/shaders/opengl/primitive.vert"))
		->attach_shader(ShaderBase("assets/shaders/opengl/primitive.frag"))
		->link_program();
}

bool GLPrimitiveApp::tick(int count) {
	process_input_callback(m_window);
	glfwPollEvents();
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	mp_shader_program->use();
	if (m_offset_state.triangle_end > m_offset_state.triangle_start) {
		glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(m_offset_state.triangle_end - m_offset_state.triangle_start), GL_UNSIGNED_INT, (void*)(m_offset_state.triangle_start * sizeof(unsigned int)));
	}

	if (m_offset_state.line_end > m_offset_state.line_start) {
		glDrawElements(GL_LINES, static_cast<unsigned int>(m_offset_state.line_end - m_offset_state.line_start), GL_UNSIGNED_INT, (void*)(m_offset_state.line_start * sizeof(unsigned int)));
	}

	if (m_offset_state.point_end > m_offset_state.point_start) {
		glDrawElements(GL_POINTS, static_cast<unsigned int>(m_offset_state.point_end - m_offset_state.point_start), GL_UNSIGNED_INT, (void*)(m_offset_state.point_start * sizeof(unsigned int)));
	}

	glfwSwapBuffers(m_window);
	return !glfwWindowShouldClose(m_window);
}

}// namespace sail::gl