#pragma once
/**
 * @file app/opengl/gl_pure_app.h
 * @author sailing-innocent
 * @date 2023-02-25
 * @brief The Pure GLPure App
 */

#include "glad/gl.h"
#include <GLFW/glfw3.h>
#include "SailIng/app.h"
#include <string>

namespace sail::ing {

class SAIL_ING_API INGGLPureApp : public INGApp {
public:
	INGGLPureApp() = default;
	INGGLPureApp(std::string _title,
				 unsigned int _resw,
				 unsigned int _resh) : m_title(_title),
									   m_resw(_resw),
									   m_resh(_resh) {
	}
	~INGGLPureApp();
	void init() override;
	bool tick(int count = 0) override;
	void terminate() override {}

protected:
	virtual void init_window(unsigned int resw, unsigned int resh);
	virtual void init_buffers(){};
	virtual void destroy_buffers(){};
	virtual void destroy_window();

	static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
	static void process_input_callback(GLFWwindow* window);

protected:
	std::string m_title = "ING_GL_Pure";
	unsigned int m_resw = 800;
	unsigned int m_resh = 600;
	GLFWwindow* m_window = nullptr;
};

}// namespace sail::ing