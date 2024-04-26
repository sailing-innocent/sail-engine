#pragma once
/**
 * @file dummy_app/pure.h
 * @author sailing-innocent
 * @date 2024-04-26
 * @brief Dummy App Pure GL implmentation
 */

#include "glad/gl.h"
#include <GLFW/glfw3.h>
#include "SailGL/config.h"
#include "SailDummy/app.h"
#include <string>

namespace sail::gl {

class SAIL_GL_API GLPureDummyApp : public DummyApp {
public:
	GLPureDummyApp() = default;
	GLPureDummyApp(std::string _title,
				   unsigned int _resw,
				   unsigned int _resh) : m_title(_title),
										 m_resw(_resw),
										 m_resh(_resh) {
	}
	virtual ~GLPureDummyApp();
	void init() override {}
	bool tick(int count = 0) override {}
	void terminate() override {}

protected:
	// virtual void init_window(unsigned int resw, unsigned int resh);
	// virtual void init_buffers(){};
	// virtual void destroy_buffers(){};
	// virtual void destroy_window();

	// static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
	// static void process_input_callback(GLFWwindow* window);

	std::string m_title = "GL_Pure";
	unsigned int m_resw = 800;
	unsigned int m_resh = 600;
	// GLFWwindow* m_window = nullptr;
};

}// namespace sail::gl