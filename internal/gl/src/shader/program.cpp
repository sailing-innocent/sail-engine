/**
 * @file program.cpp
 * @brief The Implementation of OpenGL Shader program
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailGL/shader/program.h"
#include <iostream>

namespace sail::gl {

ShaderProgram::ShaderProgram(std::string name) : m_name(name) {
	m_id = glCreateProgram();
	is_linked = false;
	is_compute = false;
}

ShaderProgram::~ShaderProgram() {
	glDeleteProgram(m_id);
}

ShaderProgram* ShaderProgram::attach_shader(ShaderBase s) {
	std::cout << "ATTACHING SHADER " << s.get_name() << " TO PROGRAM " << m_name << std::endl;
	if (!is_compute) {
		glAttachShader(m_id, s.get_shad());
		if (s.get_name() == "compute") {
			is_compute = true;
		}
		m_shaders.push_back(s.get_shad());
	} else {
		std::cout << "ERROR: TRYING TO LINK A NON COMPUTE SHADER TO COMPUTE PROGRAM" << std::endl;
	}
	return this;
}

void ShaderProgram::link_program() {
	glLinkProgram(m_id);

	if (check_compile_errors(m_id, "PROGRAM", m_name.c_str())) {
		is_linked = true;
		std::cout << "PROGRAM " << m_name << " CORRECTLY LINKED" << std::endl;
		while (!m_shaders.empty()) {
			glDeleteShader(m_shaders.back());
			m_shaders.pop_back();
		}
	} else {
		std::cout << "PROGRAM " << m_name << " NOT LINKED" << std::endl;
	}
}

void ShaderProgram::use() {
	if (is_linked) {
		glUseProgram(m_id);
	} else {
		std::cout << "ERROR: PROGRAM " << m_name << " NOT LINKED" << std::endl;
	}
}

// setters

void ShaderProgram::set_mat4(const string_view name, const glm::mat4& mat) const noexcept {
	glUniformMatrix4fv(glGetUniformLocation(m_id, name.data()), 1, GL_FALSE, glm::value_ptr(mat));
}

void ShaderProgram::set_bool(const string_view name, bool value) const noexcept {
	glUniform1i(glGetUniformLocation(m_id, name.data()), static_cast<int>(value));
}

void ShaderProgram::set_int(const string_view name, int value) const noexcept {
	glUniform1i(glGetUniformLocation(m_id, name.data()), value);
}

void ShaderProgram::set_float(const string_view name, float value) const noexcept {
	glUniform1f(glGetUniformLocation(m_id, name.data()), value);
}

void ShaderProgram::set_float4(const string_view name, float v0, float v1, float v2, float v3) const noexcept {
	glUniform4f(glGetUniformLocation(m_id, name.data()), v0, v1, v2, v3);
}

}// namespace sail::gl