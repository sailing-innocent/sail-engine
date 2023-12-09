#pragma once

/** 
 * @file terrain/window.h
 * @author sailing-innocent
 * @date 2023-08-30
 * @brief the terrain window
 */

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "../scene/camera.h"

#include <string>
#include <iostream>

namespace ing::terrain {

class SAIL_ING_API Window {
public:
	Window(int& success,
		   unsigned int width = 1600,
		   unsigned int height = 900,
		   std::string name = "Terrain Opengl");
	~Window();

	static unsigned int SCR_WIDTH;
	static unsigned int SCR_HEIGHT;

	GLFWwindow* m_window;
	GLFWwindow* get_window() { return m_window; }
	void process_input(float frame_time);
	void terminate();

	// life cycle
	bool continueLoop();
	void swap_buffers_and_poll_events();

private:
	// callbacks
	static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

private:
	std::string m_name;
};

};// namespace ing::terrain
