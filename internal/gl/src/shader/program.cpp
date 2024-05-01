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

void ShaderProgram::set_mat4(const std::string& name, const glm::mat4& mat) {
	glUniformMatrix4fv(glGetUniformLocation(m_id, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

}// namespace sail::gl