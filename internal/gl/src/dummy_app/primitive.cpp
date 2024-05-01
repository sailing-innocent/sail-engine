/**
 * @file primitive.cpp
 * @brief The Implementation of GL Primitive App
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailGL/dummy_app/primitive.h"

namespace sail::gl {
using std::make_unique;

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
}

void GLPrimitiveApp::init_shaders() {
	mp_shader_program = make_unique<ShaderProgram>("scene");
	mp_shader_program
		->attach_shader(ShaderBase("assets/shaders/learnogl/scene.vert"))
		->attach_shader(ShaderBase("assets/shaders/learnogl/scene.frag"))
		->link_program();
}

bool GLPrimitiveApp::tick(int count) {
	process_input_callback(m_window);
	glfwPollEvents();
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glfwSwapBuffers(m_window);

	return !glfwWindowShouldClose(m_window);
}

}// namespace sail::gl