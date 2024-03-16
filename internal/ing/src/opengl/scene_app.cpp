/**
 * @file app/opengl/gl_scene_app.cpp
 * @author sailing-innocent
 * @date 2023-12-15
 * @brief The GLSceneApp
 */

#include "SailIng/opengl/scene_app.h"
#include <iostream>
#include "SailIng/opengl/shader_base.h"

#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace sail::ing {

INGGLSceneApp::INGGLSceneApp() {
	mp_camera = std::make_shared<INGFlipZCamera>(
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		45.0f,
		800.0f / 600.0f,
		0.1f,
		100.0f);
}

void INGGLSceneApp::load_mesh(std::string mesh_path) {
	MeshData mesh;
	m_mesh_loader.load_obj(mesh_path, mesh);
	m_meshes.emplace_back(mesh);
}

void INGGLSceneApp::init_buffers() {
	mp_shader_program = std::make_unique<ing::ogl::ShaderProgram>("scene");
	mp_shader_program
		->attach_shader(ing::ogl::ShaderBase("assets/shaders/learnogl/scene.vert"))
		->attach_shader(ing::ogl::ShaderBase("assets/shaders/learnogl/scene.frag"))
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

bool INGGLSceneApp::tick(int count) {
	process_input_callback(m_window);
	glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	mp_shader_program->use();
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);

	glm::mat4 model = glm::mat4(1.0f);
	model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.f, 0.f));
	mp_shader_program->set_mat4("model", model);
	mp_shader_program->set_mat4("view", mp_camera->m_view_matrix);
	mp_shader_program->set_mat4("proj", mp_camera->m_proj_matrix);

	// glm::mat4 view = glm::mat4(1.0f);
	// glm::mat4 proj = glm::mat4(1.0f);
	// mp_shader_program->set_mat4("view", view);
	// mp_shader_program->set_mat4("proj", proj);

	glfwSwapBuffers(m_window);
	glfwPollEvents();

	return !glfwWindowShouldClose(m_window);
}

}// namespace sail::ing