#pragma once
/**
 * @file: include/ing/utils/gl/gl_primitive.h
 * @author: sailing-innocent
 * @date 2022-11-20
 * @brief The GL Primitive Utility Header
 */
#include "glad/gl.h"
#include "gl_shader.h"

#include <vector>
#include <span>

namespace sail::ing {

enum GLPrimitiveType {
	GL_PRIMITIVE_TRIANGLES = 0,
	GL_PRIMITIVE_LINES,
	GL_PRIMITIVE_POINTS,
	GL_PRIMITIVE_TYPES_COUNT
};

class SAIL_ING_API GLPrimitive {
public:
	GLPrimitive(GLPrimitiveType _type = GL_PRIMITIVE_TRIANGLES) : m_type(_type) {}
	virtual ~GLPrimitive() {}
	virtual void bindShader(GLShader& _shader) { m_shader = _shader; }
	virtual void appendVertices(std::span<float> _vertices);
	virtual void appendIndicies(std::span<unsigned int> _indices);
	virtual void appendPrimitive(GLPrimitive& rhs) {
		appendIndicies(rhs.indicies());
		appendVertices(rhs.vertices());
	}
	virtual std::vector<float>& vertices() { return m_vertices; }
	virtual std::vector<unsigned int>& indicies() { return m_indices; }
	unsigned int& VBO() { return m_vertex_buffer_object; }
	unsigned int& VAO() { return m_vertex_array_object; }
	unsigned int& EBO() { return m_element_array_object; }

protected:
	GLPrimitiveType m_type = GL_PRIMITIVE_TRIANGLES;
	GLShader m_shader;
	size_t m_vertices_count = 0;
	std::vector<float> m_vertices;
	std::vector<unsigned int> m_indices;
	unsigned int m_vertex_buffer_object;
	unsigned int m_vertex_array_object;
	unsigned int m_element_array_object;
};

class SAIL_ING_API GLPoint : public GLPrimitive {
public:
	GLPoint(float x = 0.0f, float y = 0.0f, float z = 0.0f) : GLPrimitive(GL_PRIMITIVE_POINTS) {
		m_vertices.resize(8);
		m_indices.resize(1);
		m_vertices_count = 1;
		m_vertices[0] = x;
		m_vertices[1] = y;
		m_vertices[2] = z;
		m_vertices[3] = 1.0f;
		m_vertices[4] = 1.0f;
		m_vertices[5] = 0.0f;
		m_vertices[6] = 0.0f;
		m_vertices[7] = 1.0f;
		m_indices[0] = 0u;
	}
	~GLPoint() {}
	void setColor(std::vector<float>& _color) {
		m_vertices[4] = _color[0];
		m_vertices[5] = _color[1];
		m_vertices[6] = _color[2];
		m_vertices[7] = _color[3];
	}

protected:
};

class SAIL_ING_API GLPointList : public GLPrimitive {
public:
	GLPointList() : GLPrimitive(GL_PRIMITIVE_POINTS) {}
	void appendPrimitive(GLPoint& p) {
		appendIndicies(p.indicies());
		appendVertices(p.vertices());
	}
	void appendPrimitive(GLPointList& p) {
		appendIndicies(p.indicies());
		appendVertices(p.vertices());
	}
};

class SAIL_ING_API GLLine : public GLPrimitive {
public:
	GLLine() : GLPrimitive(GL_PRIMITIVE_LINES) {}
	GLLine(GLPoint& p1, GLPoint& p2) {
		appendPrimitive(p1);
		appendPrimitive(p2);
	}
};

class SAIL_ING_API GLLineList : public GLPrimitive {
public:
	GLLineList() : GLPrimitive(GL_PRIMITIVE_LINES) {}
	void appendPrimitve(GLLine& rhs) {
		appendIndicies(rhs.indicies());
		appendVertices(rhs.vertices());
	}
	void appendPrimitve(GLLineList& rhs) {
		appendIndicies(rhs.indicies());
		appendVertices(rhs.vertices());
	}
};

class SAIL_ING_API GLTriangle : public GLPrimitive {
public:
	GLTriangle() = default;
	GLTriangle(GLPoint& a, GLPoint& b, GLPoint& c) : GLPrimitive() {
		appendPrimitive(a);
		appendPrimitive(b);
		appendPrimitive(c);
	}
};

class SAIL_ING_API GLTriangleList : public GLPrimitive {
public:
	GLTriangleList() = default;
	GLTriangleList(std::vector<GLTriangle>& triangles) {
		for (auto tr : triangles) {
			appendPrimitive(tr);
		}
	}
	void appendPrimitive(GLTriangle& rhs) {
		appendIndicies(rhs.indicies());
		appendVertices(rhs.vertices());
	}
	void appendPrimitive(GLTriangleList& rhs) {
		appendIndicies(rhs.indicies());
		appendVertices(rhs.vertices());
	}
};

}// namespace sail::ing
