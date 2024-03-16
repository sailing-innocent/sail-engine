#pragma once
/**
 * @file app/opengl/gl_basic_app.h
 * @author sailing-innocent
 * @date 2023-02-25 -- from gl_common.hpp 2022-11-07
 * @brief The Basic GL App
 */

#include "pure_app.h"
#include "gl_shader.h"
#include "gl_primitive.h"
#include <vector>

namespace sail::ing {

class SAIL_ING_API INGGLBasicApp : public INGGLPureApp {
public:// cons/des tructor
	INGGLBasicApp() = default;
	INGGLBasicApp(std::string _title,
				  unsigned int _resw,
				  unsigned int _resh,
				  std::string vert_shader_path = "",
				  std::string frag_shader_path = "")
		: m_title(_title), m_resw(_resw), m_resh(_resh), m_vert_shader_path(vert_shader_path), m_frag_shader_path(frag_shader_path) {
	}
	~INGGLBasicApp() {
		destroy_buffers();
		destroy_window();
	}

public:// override method
	void init() override;
	bool tick(int count = 0) override;
	void init_buffers() override;
	void init_shaders();

public:
	virtual void setVertices(std::vector<float>& _vertices);
	virtual void setIndices(std::vector<unsigned int>& _indicies);
	virtual void addPrimitive(GLPrimitive _primitive) {
		m_primitive_root.appendPrimitive(_primitive);
	}
	virtual void addTriangle(GLTriangle _triangle) {
		m_triangles.appendPrimitive(_triangle);
	}
	virtual void addTriangles(GLTriangleList _triangles) {
		m_triangles.appendPrimitive(_triangles);
	}
	virtual void addPoint(GLPoint _point) { m_points.appendPrimitive(_point); }
	virtual void addPoints(GLPointList _points) {
		m_points.appendPrimitive(_points);
	}
	virtual void addLine(GLLine _line) { m_lines.appendPrimitive(_line); }
	virtual void addLines(GLLineList _lines) {
		m_lines.appendPrimitive(_lines);
	}

protected:// overload
	std::string m_title = "ING_GL_Basic";
	unsigned int m_resw = 800;
	unsigned int m_resh = 600;

protected:
	std::string m_vert_shader_path;
	std::string m_frag_shader_path;
	GLShader m_shader;

	GLPrimitive m_primitive_root;
	GLTriangleList m_triangles;
	GLPointList m_points;
	GLLineList m_lines;
	std::vector<float> m_vertices;
	std::vector<unsigned int> m_indices;
	std::vector<GLShader> m_shaders;

	float m_start_time = 0.0f;
	size_t m_line_offset_start = 0;
	size_t m_line_offset_end = 0;
	size_t m_triangle_offset_start = 0;
	size_t m_triangle_offset_end = 0;
	size_t m_point_offset_start = 0;
	size_t m_point_offset_end = 0;
};

}// namespace sail::ing