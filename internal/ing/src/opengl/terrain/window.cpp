#include "SailIng/opengl/terrain/core/window.h"

namespace ing::terrain {

unsigned int Window::SCR_WIDTH = 1600;
unsigned int Window::SCR_HEIGHT = 900;

Window::Window(int& success, unsigned int width, unsigned int height, std::string name)
	: m_name(name) {
	SCR_WIDTH = width;
	SCR_HEIGHT = height;
	success = 1;
	// init glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	m_window = glfwCreateWindow(width, height, m_name.c_str(), NULL, NULL);
	if (m_window == nullptr) {
		glfwTerminate();
	}
	glfwMakeContextCurrent(m_window);
	glfwSetFramebufferSizeCallback(m_window, &Window::framebuffer_size_callback);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
	}
}

Window::~Window() {
	terminate();
}

bool Window::continueLoop() {
	return !glfwWindowShouldClose(m_window);
}

void Window::process_input(float frame_time) {
	if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(m_window, true);
	}
}

void Window::swap_buffers_and_poll_events() {
	glfwSwapBuffers(m_window);
	glfwPollEvents();
}

void Window::terminate() {
	glfwTerminate();
}

}// namespace ing::terrain

// callbacks

namespace ing::terrain {

void Window::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

}// namespace ing::terrain