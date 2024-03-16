#include "SailIng/opengl/terrain/scene/terrain.h"
#include "SailIng/opengl/terrain/core/scene_elements.h"
#include "SailIng/opengl/terrain/utils/plane.h"

namespace ing::terrain {

Terrain::Terrain(int grid_length) {
	glm::mat4 id;
	glm::mat4 scaleMatrix = glm::scale(id, glm::vec3(1.0, 0.0, 1.0));
	glm::mat4 positionMatrix = glm::translate(id, glm::vec3(0., 0.0, 0.));
	model_matrix = positionMatrix;

	pos_buffer = 0;
	shad = new Shader("TerrainTessShader");
	shad->attach_shader("assets/shaders/terrain/terrain.vert")
		->attach_shader("assets/shaders/terrain/terrain.frag")
		->link_program();
	m_grid_length = grid_length + (grid_length + 1) % 2;// make sure the grid length is odd
	std::cout << "m_grid_length: " << m_grid_length << std::endl;
	m_res = 4;
	initialize_plane_VAO(m_res, tileW, &plane_vao, &plane_vbo, &plane_ebo);
	m_pos_vec.resize(m_grid_length * m_grid_length);
	generate_tile_grid(glm::vec2(0.0f, 0.0f));
	update_pos_buffer(m_pos_vec);
	std::cout << "Size of m_pos_vec: " << m_pos_vec.size() << std::endl;
}

Terrain::~Terrain() {}

void Terrain::draw() {
	// get the scene elements
	SceneElements* se = DrawableObject::scene;
	// model matrix
	glm::mat4 g_world = model_matrix;
	// view projection
	glm::mat4 g_vp = se->proj_matrix * se->cam->get_view_matrix();
	shad->use();
	shad->set_mat4("g_world", g_world);
	shad->set_mat4("g_vp", g_vp);

	int n_instances = m_pos_vec.size();
	draw_vertices(n_instances);
}

void Terrain::draw_vertices(int n_instances) {
	glBindVertexArray(plane_vao);
	shad->use();
	glDrawElementsInstanced(
		GL_TRIANGLES, (m_res - 1) * (m_res - 1) * 2 * 3,
		GL_UNSIGNED_INT, 0, n_instances);
	glBindVertexArray(0);
}

void Terrain::delete_buffer() {
	glDeleteBuffers(1, &pos_buffer);
	pos_buffer = 0;
}

void Terrain::update_pos_buffer(std::vector<glm::vec2>& pos) {
	// if exists, delte
	if (pos_buffer) {
		this->delete_buffer();
	}
	// re generate a new buffer
	glGenBuffers(1, &pos_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, pos_buffer);
	glBufferData(GL_ARRAY_BUFFER, pos.size() * sizeof(glm::vec2), &pos[0], GL_STATIC_DRAW);

	glBindVertexArray(plane_vao);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);
	glVertexAttribDivisor(3, 1);
	glBindVertexArray(0);
}

void Terrain::generate_tile_grid(glm::vec2 offset) {
	float sc = tileW;

	glm::vec2 I = glm::vec2(1, 0) * sc;
	glm::vec2 J = glm::vec2(0, 1) * sc;
	for (int i = 0; i < m_grid_length; i++) {
		for (int j = 0; j < m_grid_length; j++) {
			glm::vec2 pos =
				(float)(j - m_grid_length / 2) * glm::vec2(I) +
				(float)(i - m_grid_length / 2) * glm::vec2(J);
			set_pos(i, j, pos + offset);
		}
	}
}

void Terrain::set_gui() {}

}// namespace ing::terrain