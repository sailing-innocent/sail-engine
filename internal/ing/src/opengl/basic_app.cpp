
#include "glad/gl.h"
#include <GLFW/glfw3.h>
#include "SailIng/opengl/basic_app.h"
#include <vector>

namespace sail::ing {

void INGGLBasicApp::init() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	init_window(m_resw, m_resh);
	init_shaders();
	init_buffers();
}

void INGGLBasicApp::init_shaders() {
	// printf("init shaders\n");
	if (m_vert_shader_path.length() > 0 && m_frag_shader_path.length() > 0) {
		m_shader = GLShader(m_vert_shader_path, m_frag_shader_path);
	} else {
		std::string default_vert_shader =
			"#version 450\n\
            layout(location = 0) in vec4 aPos;\n\
            layout(location = 1) in vec4 aColor;\n\
            out vec4 pColor;\n\
            void main()\n\
            {\n\
                gl_Position = aPos;\n\
                pColor      = aColor;\n\
            }";

		std::string default_frag_shader =
			"#version 450\n\
            in vec4  pColor;\n\
            out vec4 fragColor;\n\
            void main()\n\
            {\n\
                fragColor = pColor;\n\
            }";
		m_shader = GLShader();
		m_shader.compile(default_vert_shader, default_frag_shader);
	}
	// default shader
	m_shaders.push_back(m_shader);
}

void INGGLBasicApp::init_buffers() {
	// construct triangles
	m_triangle_offset_start = 0;
	m_primitive_root.appendPrimitive(m_triangles);
	m_triangle_offset_end = m_primitive_root.indicies().size();

	m_line_offset_start = m_triangle_offset_end;
	m_primitive_root.appendPrimitive(m_lines);
	m_line_offset_end = m_primitive_root.indicies().size();
	m_point_offset_start = m_line_offset_end;
	m_primitive_root.appendPrimitive(m_points);
	m_point_offset_end = m_primitive_root.indicies().size();

	// gen buffers
	glGenBuffers(1, &m_primitive_root.VBO());
	glGenBuffers(1, &m_primitive_root.EBO());
	glGenVertexArrays(1, &m_primitive_root.VAO());
	glBindVertexArray(m_primitive_root.VAO());

	// std::cout << "Binding Vertex Buffer data on " << m_primitive_root.VBO() << std::endl;
	glBindBuffer(GL_ARRAY_BUFFER, m_primitive_root.VBO());
	float* vertices = new float[m_primitive_root.vertices().size()];
	for (auto i = 0; i < m_primitive_root.vertices().size(); i++) {
		vertices[i] = m_primitive_root.vertices()[i];
		// std::cout << vertices[i] << ",";
	}
	// std::cout << "=========================" << std::endl;
	glBufferData(
		GL_ARRAY_BUFFER,
		m_primitive_root.vertices().size() * sizeof(float),
		vertices,
		GL_STATIC_DRAW);

	// std::cout << "Binding Element Buffer data on " << m_primitive_root.EBO() << std::endl;
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_primitive_root.EBO());
	unsigned int* indices = new unsigned int[m_primitive_root.indicies().size()];
	for (auto i = 0; i < m_primitive_root.indicies().size(); i++) {
		indices[i] = m_primitive_root.indicies()[i];
		// std::cout << indices[i] << ",";
	}
	// std::cout << "=========================" << std::endl;
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_primitive_root.indicies().size() * sizeof(unsigned int), indices, GL_STATIC_DRAW);

	// aPos
	glVertexAttribPointer(
		0,				  // which vertex attribute .. something like "position = 0" in shader
		4,				  // sizeof data attribute
		GL_FLOAT,		  // typeof data
		GL_FALSE,		  // normalized or not
		8 * sizeof(float),// stride
		(void*)0		  // offset
	);
	glEnableVertexAttribArray(0);
	// aColor
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(4 * sizeof(float)));
	glEnableVertexAttribArray(1);

	delete (vertices);
	delete (indices);
}

void INGGLBasicApp::setVertices(std::vector<float>& _vertices) {
	m_vertices.resize(_vertices.size());
	for (auto i = 0; i < _vertices.size(); i++) {
		m_vertices[i] = _vertices[i];
	}
}

void INGGLBasicApp::setIndices(std::vector<unsigned int>& _indices) {
	m_indices.resize(_indices.size());
	for (auto i = 0; i < _indices.size(); i++) {
		m_indices[i] = _indices[i];
	}
}

bool INGGLBasicApp::tick(int count) {
	process_input_callback(m_window);
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindVertexArray(m_primitive_root.VAO());
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_primitive_root.EBO());
	m_shaders[0].use();
	if (m_triangle_offset_end > m_triangle_offset_start) {
		glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(m_triangle_offset_end - m_triangle_offset_start), GL_UNSIGNED_INT, (void*)(m_triangle_offset_start * sizeof(unsigned int)));
	}
	if (m_line_offset_end > m_line_offset_start) {
		glDrawElements(GL_LINES, static_cast<unsigned int>(m_line_offset_end - m_line_offset_start), GL_UNSIGNED_INT, (void*)(m_line_offset_start * sizeof(unsigned int)));
	}

	if (m_point_offset_end > m_point_offset_start) {
		glDrawElements(GL_POINTS, static_cast<unsigned int>(m_point_offset_end - m_point_offset_start), GL_UNSIGNED_INT, (void*)(m_point_offset_start * sizeof(unsigned int)));
	}

	glfwSwapBuffers(m_window);
	glfwPollEvents();
	return !glfwWindowShouldClose(m_window);
}

}// namespace sail::ing