/**
 * @file gl_primitive.cpp
 * @brief The GL Primitive Utility Implementation
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailGL/util/gl_primitive.h"

namespace sail::gl {

using std::span;

void GLPrimitive::appendVertices(const span<float> _vertices) {
	size_t offset = m_vertices.size();
	m_vertices_count = _vertices.size() / 8 + m_vertices_count;
	m_vertices.resize(offset + _vertices.size());
	for (auto i = 0; i < _vertices.size(); i++) {
		m_vertices[i + offset] = _vertices[i];
	}
}

void GLPrimitive::appendIndicies(const span<unsigned int> _indices) {
	size_t offset = m_indices.size();
	m_indices.resize(offset + _indices.size());
	unsigned int u_vertices_count = static_cast<unsigned int>(m_vertices_count);
	for (auto i = 0; i < _indices.size(); i++) {
		m_indices[i + offset] = _indices[i] + u_vertices_count;
	}
}

}// namespace sail::gl