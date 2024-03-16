/**
 * @file app/opengl/gl_pure_app.cc
 * @author sailing-innocent
 * @date 2023-02-25
 * @brief the implementation for gl pure window
 */

#include "glad/gl.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "SailIng/opengl/pure_app.h"

namespace sail::ing {

void INGGLPureApp::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}
void INGGLPureApp::process_input_callback(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

INGGLPureApp::~INGGLPureApp() {
	destroy_buffers();
	destroy_window();
}

void INGGLPureApp::init() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	init_window(m_resw, m_resh);
	init_buffers();
}

void INGGLPureApp::init_window(unsigned int resw, unsigned int resh) {
	m_window = glfwCreateWindow(resw, resh, m_title.c_str(), NULL, NULL);
	if (m_window == nullptr) {
		glfwTerminate();
	}
	glfwMakeContextCurrent(m_window);
	glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
	}
}

void INGGLPureApp::destroy_window() {
	glfwTerminate();
}

bool INGGLPureApp::tick(int count) {
	process_input_callback(m_window);
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glfwSwapBuffers(m_window);
	glfwPollEvents();

	return !glfwWindowShouldClose(m_window);
}

}// namespace sail::ing