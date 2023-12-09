#pragma once
/** 
 * @file terrain/scene/terrain.h
 * @author sailing-innocent
 * @date 2023-08-30
 * @brief the terrain heightmap
 */

#include "../core/drawable_object.h"
#include "../core/shader.h"

#include <glm/glm.hpp>
#include <vector>

namespace ing::terrain {

class SAIL_ING_API Terrain : public DrawableObject {
public:
	Terrain(int grid_length);
	virtual ~Terrain();
	virtual void draw();
	virtual void set_gui();

	glm::vec2 position, eps;
	glm::mat4 model_matrix;
	float up = 0.0f;
	static const int tileW = 10. * 100.;

private:
	// gl
	unsigned int plane_vbo, plane_vao, plane_ebo;
	Shader* shad;
	unsigned int pos_buffer;
	void draw_vertices(int n_instances);
	void update_pos_buffer(std::vector<glm::vec2>& pos);
	void delete_buffer();

private:
	// resources
	std::vector<glm::vec2> m_pos_vec;
	void set_pos(int row, int col, glm::vec2 pos) { m_pos_vec[row * m_grid_length + col] = pos; }
	void generate_tile_grid(glm::vec2 offset);

private:
	// properties
	int m_res;
	int m_octaves;
	int m_grid_length;
};

}// namespace ing::terrain