#pragma once
/** 
 * @file opengl/shader/shader_program.h
 * @author sailing-innocent
 * @date 2023-12-15
 * @brief the shader program wrapper
 */
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "shader_base.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <list>

namespace sail::ing::ogl {

class SAIL_ING_API ShaderProgram {
public:
	ShaderProgram(std::string name);
	virtual ~ShaderProgram();
	ShaderProgram* attach_shader(ShaderBase s);
	void link_program();

	void use();
	// setter
	void set_mat4(const std::string& name, glm::mat4& mat);

protected:
	unsigned int m_id;
	bool is_linked, is_compute;
	std::list<unsigned int> m_shader_list;
	std::string m_name;
};

}// namespace sail::ing::ogl