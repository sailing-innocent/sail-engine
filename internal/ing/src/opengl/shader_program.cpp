#include "SailIng/opengl/shader_program.h"
#include <iostream>

namespace sail::ing::ogl {

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
		m_shader_list.push_back(s.get_shad());
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
		while (!m_shader_list.empty()) {
			glDeleteShader(m_shader_list.back());
			m_shader_list.pop_back();
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

void ShaderProgram::set_mat4(const std::string& name, glm::mat4& mat) {
	glUniformMatrix4fv(glGetUniformLocation(m_id, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

}// namespace sail::ing::ogl