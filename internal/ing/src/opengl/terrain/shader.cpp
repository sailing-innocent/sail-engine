#include "SailIng/opengl/terrain/core/shader.h"
#include <iostream>

namespace ing::terrain {

Shader::Shader(std::string name) {
	is_linked = false;
	is_compute = false;
	m_id = glCreateProgram();
	m_name = name;
}

Shader::~Shader() {
	glDeleteProgram(m_id);
}

Shader* Shader::attach_shader(ShaderBase s) {
	if (!is_compute) {
		glAttachShader(m_id, s.get_shad());
		if (s.get_name() == "compute") {
			is_compute = true;
		}
		this->m_shaders.push_back(s.get_shad());
	} else {
		std::cout << "ERROR: TRYING TO LINK A NON COMPUTE SHADER TO COMPUTE PROGRAM" << std::endl;
	}
	return this;
}

void Shader::link_program() {
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

void Shader::use() {
	if (is_linked) {
		glUseProgram(m_id);
	} else {
		std::cout << "PROGRAM " << m_name << " NOT LINKED" << std::endl;
	}
}

}// namespace ing::terrain

namespace ing::terrain {

void Shader::set_mat4(const std::string& name, glm::mat4 matrix) const {
	unsigned int mat = glGetUniformLocation(m_id, name.c_str());
	glUniformMatrix4fv(mat, 1, false, glm::value_ptr(matrix));
}

}// namespace ing::terrain