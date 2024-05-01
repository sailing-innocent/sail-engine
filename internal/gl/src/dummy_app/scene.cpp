/**
 * @file scene.cpp
 * @brief The OpenGL Scene App
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailGL/dummy_app/scene.h"
#include "SailDummy/util/loader.h"
#include "SailGL/shader/base.h"

#include <iostream>

namespace sail::gl {

using std::make_unique;
GLSceneDummyApp::GLSceneDummyApp() {
	mp_camera = make_unique<dummy::Camera>(
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		90.0f,
		800.0f / 600.0f,
		0.1f,
		100.0f);
}

void GLSceneDummyApp::load_mesh(string_view mesh_path) {
	dummy::MeshData mesh;
	dummy::load_mesh_from_obj(mesh_path, mesh);

	std::cout << "Loaded mesh: " << mesh_path << std::endl;
	std::cout << "Vertices: " << mesh.vertices.size() << std::endl;
	m_meshes.emplace_back(std::move(mesh));
}

void GLSceneDummyApp::init_buffers() {
	mp_shader_program = make_unique<ShaderProgram>("scene");
	mp_shader_program
		->attach_shader(ShaderBase("assets/shaders/learnogl/scene.vert"))
		->attach_shader(ShaderBase("assets/shaders/learnogl/scene.frag"))
		->link_program();

	// combine vertices
	for (auto& mesh : m_meshes) {
		m_vertices.insert(m_vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
		m_indices.insert(m_indices.end(), mesh.indices.begin(), mesh.indices.end());
	}

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(float), m_vertices.data(), GL_STATIC_DRAW);
	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(unsigned int), m_indices.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
}

bool GLSceneDummyApp::tick(int count) {
	process_input_callback(m_window);
	glfwPollEvents();

	glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	mp_shader_program->use();
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);

	glm::mat4 model = glm::mat4(1.0f);
	model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.f, 0.f));
	mp_shader_program->set_mat4("model", model);
	mp_shader_program->set_mat4("view", mp_camera->view_matrix());
	mp_shader_program->set_mat4("proj", mp_camera->proj_matrix());

	glfwSwapBuffers(m_window);

	return !glfwWindowShouldClose(m_window);
}

}// namespace sail::gl