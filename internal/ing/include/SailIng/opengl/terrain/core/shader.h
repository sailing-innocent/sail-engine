#pragma once
/** 
 * @file terrain/shader.h
 * @author sailing-innocent
 * @date 2023-08-30
 * @brief the terrain tesslation shader
 */

#include "shader_base.h"

#include <glad/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <list>

namespace ing::terrain {

class SAIL_ING_API Shader {
public:
	Shader(std::string name);
	virtual ~Shader();
	Shader* attach_shader(ShaderBase s);
	void link_program();

	void use();
	// void setBool(const std::string& name, bool value) const;
	// void setInt(const std::string& name, int value) const;
	// void setFloat(const std::string& name, float value) const;
	// void setVec2(const std::string& name, glm::vec2 vector) const;
	// void setVec3(const std::string& name, glm::vec3 vector) const;
	// void setVec4(const std::string& name, glm::vec4 vector) const;
	void set_mat4(const std::string& name, glm::mat4 matrix) const;
	// void setSampler2D(const std::string& name, unsigned int texture, int id) const;
	// void setSampler3D(const std::string& name, unsigned int texture, int id) const;

protected:
	unsigned int m_id;
	bool is_linked, is_compute;
	std::list<unsigned int> m_shaders;
	std::string m_name;
};

}// namespace ing::terrain