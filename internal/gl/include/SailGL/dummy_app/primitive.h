#pragma once

/**
 * @file primitive.h
 * @brief The GL Primitive App
 * @author sailing-innocent
 * @date 2022-11-07
 */

#include "pure.h"
#include "SailGL/shader/program.h"
#include "SailGL/util/gl_primitive.h"
#include <memory>
#include <string>

namespace sail::gl {
using std::string;
using std::unique_ptr;

class SAIL_GL_API GLPrimitiveApp : public GLPureDummyApp {
	struct OffsetState {
		int line_start = 0;
		int line_end = 0;
		int point_start = 0;
		int point_end = 0;
		int triangle_start = 0;
		int triangle_end = 0;
	};
	OffsetState m_offset_state;

public:
	GLPrimitiveApp() = default;
	GLPrimitiveApp(
		string _title,
		unsigned int _resw,
		unsigned int _resh,
		string vert_shader_path = "",
		string frag_shader_path = "") : m_title(_title),
										m_resw(_resw),
										m_resh(_resh),
										m_vert_shader_path(vert_shader_path),
										m_frag_shader_path(frag_shader_path) {}

	~GLPrimitiveApp() {
		destroy_buffers();
		destroy_window();
	}

	void init() override;
	bool tick(int count = 0) override;
	void init_buffers() override;
	void init_shaders();

	// primitive loader

protected:
	std::string m_title = "GL Primitive App";
	unsigned int m_resw = 800;
	unsigned int m_resh = 600;
	std::string m_vert_shader_path;
	std::string m_frag_shader_path;

	unique_ptr<ShaderProgram> mp_shader_program;
	unsigned int VAO, VBO, EBO;
	unique_ptr<GLPrimitive> mp_primitive_root;
	unique_ptr<GLTriangleList> mp_triangles;
	unique_ptr<GLPointList> mp_points;
	unique_ptr<GLLineList> mp_lines;

	std::vector<float> m_vertices;
	std::vector<unsigned int> m_indices;
};

}// namespace sail::gl