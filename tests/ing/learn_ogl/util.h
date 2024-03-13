#pragma once
#include <glad/gl.h>
#include <GLFW/glfw3.h>

namespace sail::ing {
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
}// namespace sail::ing