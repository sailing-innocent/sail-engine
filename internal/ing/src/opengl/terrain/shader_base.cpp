#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "SailIng/opengl/terrain/core/shader_base.h"

#include <iostream>
#include <fstream>
#include <sstream>

namespace ing::terrain {

ShaderBase::ShaderBase(const char* shader_path) {
	auto path = std::string(shader_path);
	std::string shader_string = load_shader_from_file(shader_path);
	const char* shader_code = shader_string.c_str();

	m_shader_type = get_shader_type(shader_path);
	m_shad = glCreateShader(m_shader_type.m_type);
	glShaderSource(m_shad, 1, &shader_code, NULL);
	glCompileShader(m_shad);
	check_compile_errors(
		m_shad, m_shader_type.m_name, get_shader_name(shader_path).c_str());
}

bool check_compile_errors(unsigned int shader, std::string type_name, std::string shader_name) {
	int success;
	char infoLog[1024];
	if (type_name != "PROGRAM") {
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR: SHADER" << shader_name
					  << "COMPILATION ERROR of type: " << type_name << "\n"
					  << infoLog << "\n -- --------------------------------------------------- -- "
					  << std::endl;
		}
	} else {
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type_name << "\n"
					  << infoLog << "\n -- --------------------------------------------------- -- "
					  << std::endl;
		}
	}

	if (success) {
		std::cout << type_name + " SHADER SUCCESSFULLY COMPILED AND/OR LINKED!"
				  << std::endl;
	}
	return success;
}

ShaderBase::~ShaderBase() {}

std::string ShaderBase::load_shader_from_file(const char* shader_path) {
	std::string shader_code;
	std::ifstream shader_file;
	shader_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try {
		shader_file.open(shader_path);
		std::stringstream shader_stream;
		shader_stream << shader_file.rdbuf();
		shader_file.close();
		shader_code = shader_stream.str();
	} catch (std::ifstream::failure e) {
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ"
				  << get_shader_name(shader_path) << std::endl;
	}
	return shader_code;
}

std::string get_shader_name(const char* path) {
	std::string pathstr = std::string(path);
	const size_t last_slash_idx = pathstr.find_last_of("/");
	if (std::string::npos != last_slash_idx) {
		pathstr.erase(0, last_slash_idx + 1);
	}
	return pathstr;
}

ShaderType get_shader_type(const char* path) {
	std::string name = get_shader_name(path);
	const size_t last_slash_idx = name.find_last_of(".");
	if (std::string::npos != last_slash_idx) {
		name.erase(0, last_slash_idx + 1);
	}
	if (name == "vert")
		return ShaderType(GL_VERTEX_SHADER, "VERTEX");
	if (name == "frag")
		return ShaderType(GL_FRAGMENT_SHADER, "FRAGMENT");
	if (name == "tes")
		return ShaderType(GL_TESS_EVALUATION_SHADER, "TESS_EVALUATION");
	if (name == "tcs")
		return ShaderType(GL_TESS_CONTROL_SHADER, "TESS_CONTROL");
	if (name == "geom")
		return ShaderType(GL_GEOMETRY_SHADER, "GEOMETRY");
	if (name == "comp")
		return ShaderType(GL_COMPUTE_SHADER, "COMPUTE");
}

}// namespace ing::terrain