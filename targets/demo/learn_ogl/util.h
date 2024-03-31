#pragma once
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <string>

namespace sail::demo {
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

struct GLShader {
	unsigned int id;// the shader program id
	GLShader() = default;
	GLShader(std::string& vertexPath, std::string& fragmentPath);
	GLShader(const char* vertexPath, const char* fragmentPath);
	void use();// use/activate the shader
	GLShader& operator=(const GLShader& rhs) {
		id = rhs.id;
		return *this;
	}
	void compile(const std::string& vertex_code, const std::string& fragment_code);
	void set_bool(const std::string& name, bool value) const;
	void set_int(const std::string& name, int value) const;
	void set_float(const std::string& name, float value) const;
	void set_float4(const std::string& name, float v0, float v1, float v2, float v3) const;
	void set_mat4(const std::string& name, float* value_ptr);
};

}// namespace sail::demo