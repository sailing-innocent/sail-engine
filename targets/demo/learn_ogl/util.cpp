#include "util.h"
#include "glad/gl.h"
#include <glfw/glfw3.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

namespace sail::ing::test {

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	// make sure the viewport matches the new window dimensions; note that width and
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react
// accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

GLShader::GLShader(const char* vertex_path, const char* fragment_path) {
	std::cout << "is constructing GL shader through file: " << vertex_path << " and " << fragment_path << std::endl;
	// read vertex and fragment shader source from path
	std::string vertex_code;
	std::string fragment_code;
	std::ifstream v_shader_file;
	std::ifstream f_shader_file;

	v_shader_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	f_shader_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try {
		v_shader_file.open(vertex_path);
		f_shader_file.open(fragment_path);

		std::stringstream v_shader_stream, f_shader_stream;
		v_shader_stream << v_shader_file.rdbuf();
		f_shader_stream << f_shader_file.rdbuf();

		vertex_code = v_shader_stream.str();
		fragment_code = f_shader_stream.str();
	} catch (std::ifstream::failure e) {
		std::cout << "ERROR::SHADER::FILE NOT SUCCESSFULLY_READ: " << e.what() << std::endl;
	}
	compile(vertex_code, fragment_code);
}

void GLShader::compile(const std::string& vertex_code, const std::string& fragment_code) {
	const char* v_shader_code = vertex_code.c_str();
	const char* f_shader_code = fragment_code.c_str();

	unsigned int vertex, fragment;
	// compile vertex shader
	int success;
	char infolog[512];

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &v_shader_code, NULL);
	glCompileShader(vertex);
	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);

	if (!success) {
		glGetShaderInfoLog(vertex, 512, NULL, infolog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n"
				  << infolog << std::endl;
	}

	// compile fragment shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &f_shader_code, NULL);
	glCompileShader(fragment);
	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragment, 512, NULL, infolog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n"
				  << infolog << std::endl;
	};

	// link program
	id = glCreateProgram();
	glAttachShader(id, vertex);
	glAttachShader(id, fragment);
	glLinkProgram(id);
	glGetProgramiv(id, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(id, 512, NULL, infolog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
				  << infolog << std::endl;
	}
	std::cout << "PROGRME_COMPILE_SUCCESS" << std::endl;
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}

GLShader::GLShader(std::string& vertex_path, std::string& fragment_path)
	: GLShader(vertex_path.c_str(), fragment_path.c_str()) {
}

void GLShader::use() {
	glUseProgram(id);
}

void GLShader::set_bool(const std::string& name, bool value) const {
	glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value);
}

void GLShader::set_int(const std::string& name, int value) const {
	glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value);
}
void GLShader::set_float(const std::string& name, float value) const {
	glUniform1f(glGetUniformLocation(id, name.c_str()), value);
}

void GLShader::set_float4(const std::string& name, float v0, float v1, float v2, float v3) const {
	glUniform4f(glGetUniformLocation(id, name.c_str()), v0, v1, v2, v3);
}

void GLShader::set_mat4(const std::string& name, float* value_ptr) {
	glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, value_ptr);
}

}// namespace sail::ing::test