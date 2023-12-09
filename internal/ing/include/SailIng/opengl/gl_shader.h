#pragma once
/**
 * @file app/opengl/gl_shader.h
 * @author sailing-innocent
 * @date 2023-10-21
 * @brief The GL Shader
 */
#include "SailBase/config.h"
#include <GLFW/glfw3.h>
#include <string>

namespace sail::ing {

struct SAIL_ING_API GLShader {
	unsigned int id;// the shader program id
	GLShader() = default;
	GLShader(std::string& vertexPath, std::string& fragmentPath);
	GLShader(const char* vertexPath, const char* fragmentPath);
	void use();// use/activate the shader
	GLShader& operator=(const GLShader& rhs) = default;
	void compile(const std::string& vertex_code, const std::string& fragment_code);
	void set_bool(const std::string& name, bool value) const;
	void set_int(const std::string& name, int value) const;
	void set_float(const std::string& name, float value) const;
	void set_float4(const std::string& name, float v0, float v1, float v2, float v3) const;
	void set_mat4(const std::string& name, float* value_ptr);
};

}// namespace sail::ing